#!/usr/bin/env python3
"""EXP-014: Medição numérica do Delta real (Base -> Instruct)
Testa a hipótese de Multi-Fine-Tune (Base + Δ_1bit e Base + Δ_2bit).
Mantém RSS sob limite estrito (< 4.0 GiB) processando tensores com safe_open.
"""
from __future__ import annotations

import gc
import json
import os
import resource
import time
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "experiments" / "logs" / "crom-infer" / "exp014_bitdelta.json"

BASE_ID = "ibm-granite/granite-3.1-1b-a400m-base"
INST_ID = "ibm-granite/granite-3.1-1b-a400m-instruct"

VAL_TEXT = (
    "A Mixture-of-Experts model routes each token to a few experts, which is why reading the "
    "right experts from disk matters more than loading everything. Uma tabela hash mapeia chaves "
    "para baldes usando uma função de hash, e colisões são resolvidas por encadeamento ou "
    "endereçamento aberto. Sparse deltas between experts can be stored on a SATA SSD and pulled "
    "on demand, since reads of 1 MiB saturate the bandwidth. O codebook transforma multiplicação "
    "em busca de tabela na cache L1/L2 do processador, evitando FLOPs no caminho crítico."
)

PROMPT2 = "What is a Mixture-of-Experts neural network? Answer concisely in 3 sentences."
G = 64


def get_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def compute_ppl(model, tok) -> tuple[float, float]:
    ids = tok(VAL_TEXT, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(ids).logits[:, :-1].float()
    nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1)
    )
    return nll.item(), float(nll.exp().item())


def generate_prompt2(model, tok, max_new_tokens=64) -> str:
    messages = [{"role": "user", "content": PROMPT2}]
    # Se o tokenizador não tiver chat template (como o Base), usa prompt simples
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_str = f"User: {PROMPT2}\nAssistant: "
    
    input_ids = tok(prompt_str, return_tensors="pt").input_ids
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=tok.eos_token_id or 0,
        )
    gen_text = tok.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return gen_text.strip()


def quant_delta_1bit(diff: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Quantização de 1 bit por peso por grupo de 64: sign(Δ) * mean(|Δ|)."""
    orig_shape = diff.shape
    diff_g = diff.float().view(-1, G)
    alpha = diff_g.abs().mean(dim=1, keepdim=True).half().float()
    signs = torch.where(diff_g >= 0, 1.0, -1.0)
    recon = (signs * alpha).view(orig_shape).to(diff.dtype)
    num_groups = diff_g.shape[0]
    # 64 bits para signs (8 bytes) + 2 bytes FP16 para alpha = 10 bytes por grupo de 64
    bytes_size = num_groups * 10
    return recon, bytes_size


def quant_delta_2bit(diff: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Quantização de 2 bits simétrica por grupo de 64: q in [-2..1] ou [-1,0,1]."""
    orig_shape = diff.shape
    diff_g = diff.float().view(-1, G)
    scale = (diff_g.abs().amax(dim=1, keepdim=True) / 1.5).clamp(min=1e-12).half().float()
    q = torch.clamp(torch.round(diff_g / scale), -2.0, 1.0)
    recon = (q * scale).view(orig_shape).to(diff.dtype)
    num_groups = diff_g.shape[0]
    # 128 bits para códigos 2-bit (16 bytes) + 2 bytes FP16 scale = 18 bytes por grupo de 64
    bytes_size = num_groups * 18
    return recon, bytes_size


def main():
    torch.set_num_threads(6)
    print("=== EXP-014: BitDelta Base -> Instruct ===", flush=True)
    print(f"RSS Inicial: {get_rss_mb():.1f} MB", flush=True)

    # 1. Carrega tokenizador do Instruct (que contém todos os tokens e o chat template)
    tok_inst = AutoTokenizer.from_pretrained(INST_ID, local_files_only=True)
    tok_base = AutoTokenizer.from_pretrained(BASE_ID, local_files_only=True)

    # Localiza arquivo model.safetensors do Instruct
    from huggingface_hub import scan_cache_dir
    cache = scan_cache_dir()
    inst_safetensors = None
    for r in cache.repos:
        if r.repo_id == INST_ID:
            for rev in r.revisions:
                candidate = os.path.join(rev.snapshot_path, "model.safetensors")
                if os.path.exists(candidate):
                    inst_safetensors = candidate
                    break
    assert inst_safetensors, "Arquivo model.safetensors do Instruct não encontrado no cache!"

    # 2. Carrega o Modelo Base na memória CPU (bfloat16)
    print(f"Carregando {BASE_ID} na CPU (bfloat16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True
    )
    model.eval()
    print(f"Modelo Base carregado. RSS={get_rss_mb():.1f} MB", flush=True)

    # a) Avalia Base Puro
    print("Avaliando PPL e geração do Base puro...", flush=True)
    nll_base, ppl_base = compute_ppl(model, tok_base)
    gen_base = generate_prompt2(model, tok_base)
    print(f"  [Base Puro] NLL={nll_base:.4f} PPL={ppl_base:.3f}")
    print(f"  [Base Puro Resposta Prompt 2]:\n  {gen_base[:180]}...\n", flush=True)

    # 3. Medição L2 das diferenças por componente
    print("Analisando diferenças W_instruct - W_base tensor por tensor via safe_open...", flush=True)
    l2_metrics = {
        "embed_tokens": {"l2_diff": 0.0, "l2_inst": 0.0},
        "norms": {"l2_diff": 0.0, "l2_inst": 0.0},
        "routers": {"l2_diff": 0.0, "l2_inst": 0.0},
        "attn": {"l2_diff": 0.0, "l2_inst": 0.0},
        "experts": {"l2_diff": 0.0, "l2_inst": 0.0},
    }

    base_params = dict(model.named_parameters())

    with safe_open(inst_safetensors, framework="pt", device="cpu") as sf:
        for name, p_base in base_params.items():
            if name not in sf.keys():
                continue
            p_inst = sf.get_tensor(name)
            
            # Categoria
            if "embed_tokens" in name:
                cat = "embed_tokens"
                # Base tem 49152 e Instruct tem 49155
                d = (p_inst[:49152] - p_base).float()
                l2_metrics[cat]["l2_diff"] += d.norm().item() ** 2
                l2_metrics[cat]["l2_inst"] += p_inst[:49152].float().norm().item() ** 2
                continue
            elif "norm" in name:
                cat = "norms"
            elif "block_sparse_moe.gate" in name:
                cat = "routers"
            elif any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj")):
                cat = "attn"
            elif any(k in name for k in ("input_linear", "output_linear")):
                cat = "experts"
            else:
                cat = "norms"

            d = (p_inst - p_base).float()
            l2_metrics[cat]["l2_diff"] += d.norm().item() ** 2
            l2_metrics[cat]["l2_inst"] += p_inst.float().norm().item() ** 2
            del p_inst, d

    l2_summary = {}
    for cat, vals in l2_metrics.items():
        rel_l2 = (vals["l2_diff"] ** 0.5) / (vals["l2_inst"] ** 0.5 + 1e-12)
        l2_summary[cat] = round(rel_l2, 6)
        print(f"  Norma L2 Relativa ({cat}): {rel_l2:.6f}")

    # b) Avalia Instruct Puro (atualiza model in-place com p_inst)
    print("\nAplicando Instruct puro no modelo...", flush=True)
    # Expande embedding para 49155
    with safe_open(inst_safetensors, framework="pt", device="cpu") as sf:
        new_embed = sf.get_tensor("model.embed_tokens.weight")
        model.resize_token_embeddings(new_embed.shape[0])
        model.model.embed_tokens.weight.data.copy_(new_embed)
        if hasattr(model, "lm_head") and model.lm_head is not None:
            model.lm_head.weight.data.copy_(new_embed)

        for name, p in model.named_parameters():
            if "embed_tokens" in name or "lm_head" in name:
                continue
            if name in sf.keys():
                p.data.copy_(sf.get_tensor(name))
    
    nll_inst, ppl_inst = compute_ppl(model, tok_inst)
    gen_inst = generate_prompt2(model, tok_inst)
    print(f"  [Instruct Puro] NLL={nll_inst:.4f} PPL={ppl_inst:.3f}")
    print(f"  [Instruct Puro Resposta Prompt 2]:\n  {gen_inst[:180]}...\n", flush=True)

    # c) Avalia Base + Δ_1bit
    print("Construindo e aplicando Base + Δ_1bit...", flush=True)
    # Recarrega Base embedding base + 3 tokens extras do Instruct
    bytes_delta_1bit = 0
    # Patches não-quantizados:
    # embed: 3 tokens extras [3, 1024] em FP16 = 6144 B
    bytes_delta_1bit += 3 * 1024 * 2
    # norms: 49 * 1024 * 4 B = 200704 B
    bytes_delta_1bit += 49 * 1024 * 4
    # routers: 24 * 32 * 1024 * 4 B = 3145728 B
    bytes_delta_1bit += 24 * 32 * 1024 * 4

    # Recarrega base original tensor a tensor para attn e experts
    from transformers import AutoConfig
    base_shards = []
    for r in cache.repos:
        if r.repo_id == BASE_ID:
            for rev in r.revisions:
                base_shards = [os.path.join(rev.snapshot_path, f) for f in os.listdir(rev.snapshot_path) if f.endswith(".safetensors")]
                break
    
    base_sf_map = {}
    for shard in base_shards:
        with safe_open(shard, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                base_sf_map[k] = shard

    def get_base_tensor(k):
        with safe_open(base_sf_map[k], framework="pt", device="cpu") as sf:
            return sf.get_tensor(k)

    with safe_open(inst_safetensors, framework="pt", device="cpu") as sf:
        for name, p in model.named_parameters():
            if "embed_tokens" in name or "lm_head" in name:
                continue
            if "block_sparse_moe.gate" in name or "norm" in name:
                # Mantém patch FP32 do Instruct
                p.data.copy_(sf.get_tensor(name))
            else:
                # Matriz linear (attn ou expert): quantiza delta em 1-bit
                w_base = get_base_tensor(name)
                w_inst = sf.get_tensor(name)
                diff = w_inst - w_base
                recon_delta, b_sz = quant_delta_1bit(diff)
                bytes_delta_1bit += b_sz
                p.data.copy_(w_base + recon_delta)
                del w_base, w_inst, diff, recon_delta

    nll_1bit, ppl_1bit = compute_ppl(model, tok_inst)
    gen_1bit = generate_prompt2(model, tok_inst)
    print(f"  [Base + Δ_1bit] NLL={nll_1bit:.4f} PPL={ppl_1bit:.3f} (Tamanho estimado: {bytes_delta_1bit/1e6:.1f} MB)")
    print(f"  [Base + Δ_1bit Resposta Prompt 2]:\n  {gen_1bit[:180]}...\n", flush=True)

    # d) Avalia Base + Δ_2bit
    print("Construindo e aplicando Base + Δ_2bit...", flush=True)
    bytes_delta_2bit = 3 * 1024 * 2 + 49 * 1024 * 4 + 24 * 32 * 1024 * 4
    with safe_open(inst_safetensors, framework="pt", device="cpu") as sf:
        for name, p in model.named_parameters():
            if "embed_tokens" in name or "lm_head" in name or "block_sparse_moe.gate" in name or "norm" in name:
                continue
            w_base = get_base_tensor(name)
            w_inst = sf.get_tensor(name)
            diff = w_inst - w_base
            recon_delta, b_sz = quant_delta_2bit(diff)
            bytes_delta_2bit += b_sz
            p.data.copy_(w_base + recon_delta)
            del w_base, w_inst, diff, recon_delta

    nll_2bit, ppl_2bit = compute_ppl(model, tok_inst)
    gen_2bit = generate_prompt2(model, tok_inst)
    print(f"  [Base + Δ_2bit] NLL={nll_2bit:.4f} PPL={ppl_2bit:.3f} (Tamanho estimado: {bytes_delta_2bit/1e6:.1f} MB)")
    print(f"  [Base + Δ_2bit Resposta Prompt 2]:\n  {gen_2bit[:180]}...\n", flush=True)

    results = {
        "l2_relative_diff": l2_summary,
        "results": {
            "base_puro": {"nll": round(nll_base, 4), "ppl": round(ppl_base, 3), "gen": gen_base},
            "instruct_puro": {"nll": round(nll_inst, 4), "ppl": round(ppl_inst, 3), "gen": gen_inst},
            "base_delta_1bit": {
                "nll": round(nll_1bit, 4), "ppl": round(ppl_1bit, 3),
                "delta_size_mb": round(bytes_delta_1bit / 1e6, 2), "gen": gen_1bit
            },
            "base_delta_2bit": {
                "nll": round(nll_2bit, 4), "ppl": round(ppl_2bit, 3),
                "delta_size_mb": round(bytes_delta_2bit / 1e6, 2), "gen": gen_2bit
            },
        },
        "full_model_mb": 829.47,
        "savings_1bit_pct": round((1.0 - (bytes_delta_1bit / 1e6) / 829.47) * 100, 1),
        "savings_2bit_pct": round((1.0 - (bytes_delta_2bit / 1e6) / 829.47) * 100, 1),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Resultados salvos em: {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
