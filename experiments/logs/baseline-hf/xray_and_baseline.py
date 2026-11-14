#!/usr/bin/env python3
"""EXP-003: raio-x anatomico do granite-3.1-1b-a400m-instruct (bf16, CPU)
+ baseline de texto/PPL como referencia exata vs. Ollama Q8_0 (EXP-002).

Caminhos testados numericamente:
  A/C (delta esparso): std(W) vs std(Delta), cosseno expert~Base, esparsidade de Delta.
  B (codebook): erro relativo K=256 quantizando W direto vs quantizando Delta.
"""
import json
import resource
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-instruct"
OUT_JSON = "hf_baseline.json"
XRAY_LAYERS = [0, 11, 23]
PROMPTS = [
    "Explique em 3 frases curtas como funciona uma tabela hash.",
    "What is a Mixture-of-Experts neural network? Answer concisely in 3 sentences.",
    "Escreva apenas uma função em C que soma dois vetores float de tamanho n. Sem texto extra.",
    "Resolva passo a passo e dê o resultado final: Se um disco lê a 250 MB/s e cada especialista "
    "tem 10 MB, quantos milissegundos leva para ler 8 especialistas?",
]
VAL_TEXT = (
    "A Mixture-of-Experts model routes each token to a few experts, which is why reading the "
    "right experts from disk matters more than loading everything. Uma tabela hash mapeia chaves "
    "para baldes usando uma função de hash, e colisões são resolvidas por encadeamento ou "
    "endereçamento aberto. Sparse deltas between experts can be stored on a SATA SSD and pulled "
    "on demand, since reads of 1 MiB saturate the bandwidth. O codebook transforma multiplicação "
    "em busca de tabela na cache L1/L2 do processador, evitando FLOPs no caminho crítico."
)


def mb(n_params: int) -> float:
    return n_params * 2 / 1024**2  # bf16


def xray_params(model) -> dict:
    buckets = {"embeddings_lm_head": 0, "attention_norms": 0, "routers_gates": 0,
               "experts_moe": 0, "outros": 0}

    def bucket(n: str) -> str:
        if "block_sparse_moe" in n:
            return "experts_moe" if "router" not in n else "routers_gates"
        if "embed_tokens" in n or "lm_head" in n:
            return "embeddings_lm_head"
        if "self_attn" in n or "norm" in n or "layernorm" in n or "residual_multiplier" in n:
            return "attention_norms"
        if "router" in n or "gate" in n:
            return "routers_gates"
        return "outros"

    for n, p in model.named_parameters():
        buckets[bucket(n)] += p.numel()
    total = sum(buckets.values())
    print("\n== RAIO-X 1: PARAMETROS E MEMORIA (bf16) ==")
    for k, v in buckets.items():
        print(f"  {k:20s}: {v:12,d} params | {mb(v):9.1f} MB")
    print(f"  {'TOTAL':20s}: {total:12,d} params | {mb(total):9.1f} MB")

    print("\n  Tensores da camada 0:")
    for n, p in model.named_parameters():
        if n.startswith("model.layers.0."):
            print(f"    {n}: {tuple(p.shape)}")
    return {"buckets": buckets, "total": total}


def expert_stacks(layer_mod) -> dict:
    out = {}
    for n, p in layer_mod.named_parameters():
        if "router" in n:
            continue
        if p.ndim >= 2 and p.shape[0] == 32:
            kind = "input_linear" if "input_linear" in n else ("output_linear" if "output_linear" in n else "outro")
            out[kind] = (n, p)
    return out


def similarity_metrics(W: torch.Tensor) -> dict:
    W = W.float()
    base = W.mean(0)
    delta = W - base
    flat_w = W.view(32, -1)
    flat_b = base.view(-1)
    cos = torch.nn.functional.cosine_similarity(flat_w, flat_b, dim=1).mean().item()
    sd, sw = delta.std().item(), W.std().item()
    fracs = {}
    for t in (0.25, 0.5, 1.0):
        fracs[f"|Delta|<{t}*std"] = (delta.abs() < t * sd).float().mean().item()
    return {"std_W": sw, "std_Delta": sd, "razao_std_W/std_Delta": sw / sd,
            "cosseno_medio_expert_Base": cos, "esparsidade_Delta": fracs}


def xray_similarity(model) -> dict:
    print("\n== RAIO-X 2: SIMILARIDADE MoE (Base = media dos 32 experts) ==")
    per_layer, acc = {}, {"input_linear": [], "output_linear": []}
    n_layers = len(model.model.layers)
    for i, layer in enumerate(model.model.layers):
        stacks = expert_stacks(layer)
        for kind, (name, p) in stacks.items():
            m = similarity_metrics(p)
            per_layer[f"L{i}.{kind}"] = m
            acc[kind].append(m)
    for i in XRAY_LAYERS:
        for kind in ("input_linear", "output_linear"):
            m = per_layer[f"L{i}.{kind}"]
            print(f"  L{i:2d} {kind:13s}: std(W)={m['std_W']:.5f} std(Delta)={m['std_Delta']:.5f} "
                  f"razao={m['razao_std_W/std_Delta']:.2f} cos={m['cosseno_medio_expert_Base']:.4f} "
                  f"espars={ {k: round(v, 3) for k, v in m['esparsidade_Delta'].items()} }")
    glob = {}
    for kind in ("input_linear", "output_linear"):
        glob[kind] = {k: float(np.mean([m[k] for m in acc[kind]])) if not isinstance(acc[kind][0][k], dict)
                      else {kk: float(np.mean([m[k][kk] for m in acc[kind]])) for kk in acc[kind][0][k]}
                      for k in ("std_W", "std_Delta", "razao_std_W/std_Delta", "cosseno_medio_expert_Base", "esparsidade_Delta")}
    print("  MEDIA GLOBAL (todas as camadas):")
    for kind, g in glob.items():
        print(f"    {kind:13s}: std(W)={g['std_W']:.5f} std(Delta)={g['std_Delta']:.5f} "
              f"razao={g['razao_std_W/std_Delta']:.2f} cos={g['cosseno_medio_expert_Base']:.4f} "
              f"espars={ {k: round(v, 3) for k, v in g['esparsidade_Delta'].items()} }")
    return {"por_camada": {k: {kk: (vv if not isinstance(vv, dict) else {k2: round(v2, 5) for k2, v2 in vv.items()})
                               for kk, vv in m.items()} for k, m in per_layer.items()},
            "media_global": glob, "n_camadas": n_layers}


def kmeans(X: np.ndarray, k: int = 256, iters: int = 12, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        codes = np.empty(len(X), dtype=np.int32)
        for s in range(0, len(X), 20000):
            chunk = X[s:s + 20000]
            d = ((chunk[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            codes[s:s + 20000] = d.argmin(1)
        for j in range(k):
            sel = X[codes == j]
            if len(sel):
                C[j] = sel.mean(0)
    return C, codes


def codebook_test(model) -> dict:
    print("\n== RAIO-X 3: CODEBOOK K=256 — W direto vs Delta (camada 11, d=4, 100k micro-vetores) ==")
    name, p = expert_stacks(model.model.layers[11])["input_linear"]
    W = p.detach().float()
    n_exp, hid, ff = W.shape
    rng = np.random.default_rng(7)
    idx = rng.choice(n_exp * hid * ff // 4, 100_000, replace=False)
    Wrows = W.view(-1, 4).numpy()[idx]
    base_flat = base = W.mean(0).view(-1, 4).numpy()  # [hid*ff//4, 4]
    groups = hid * ff // 4
    base_rows = base_flat[idx % groups]  # base do expert da linha amostrada
    Drows = Wrows - base_rows

    Cw, code_w = kmeans(Wrows)
    err_w = np.linalg.norm(Wrows - Cw[code_w]) / np.linalg.norm(Wrows)

    Cd, code_d = kmeans(Drows)
    W_hat = base_rows + Cd[code_d]
    err_d = np.linalg.norm(Wrows - W_hat) / np.linalg.norm(Wrows)

    print(f"  tensor: {name} {tuple(p.shape)}")
    print(f"  ||W|| ref = {np.linalg.norm(Wrows):.2f}")
    print(f"  erro relativo W  direto (K=256): {err_w:.4f}")
    print(f"  erro relativo Delta→W (K=256):   {err_d:.4f}")
    print(f"  ganho (err_W/err_Delta): {err_w / err_d:.2f}x")
    return {"tensor": name, "shape": list(p.shape), "d": 4, "k": 256, "amostra": 100_000,
            "erro_relativo_W_direto": float(err_w), "erro_relativo_via_Delta": float(err_d),
            "ganho": float(err_w / err_d)}


def perplexity(model, tok) -> float:
    ids = tok(VAL_TEXT, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(ids).logits[:, :-1].float()
    nll = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), ids[:, 1:].view(-1))
    ppl = nll.exp().item()
    print(f"\n== RAIO-X 4: PERPLEXIDADE (bf16, CPU, {ids.shape[1]} tokens de validação) ==")
    print(f"  NLL media = {nll.item():.4f} | PPL = {ppl:.3f}")
    return {"n_tokens": ids.shape[1], "nll": nll.item(), "ppl": ppl}


def prompts_baseline(model, tok) -> list:
    print(f"\n== RAIO-X 5: 4 PROMPTS PADRAO (greedy, max_new_tokens=128, threads={torch.get_num_threads()}) ==")
    out = []
    for i, p in enumerate(PROMPTS, 1):
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").input_ids
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(ids, do_sample=False, max_new_tokens=128)
        dt = time.perf_counter() - t0
        n = gen.shape[1] - ids.shape[1]
        resp = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        tok_s = n / dt
        print(f"\n  === prompt {i} (PyTorch bf16 CPU) ===")
        print("  PROMPT:", p)
        print("  RESPOSTA:", resp.strip())
        print(f"  tokens gerados={n} | tempo={dt:.1f}s | tok/s={tok_s:.2f}")
        out.append({"n": i, "prompt": p, "response": resp.strip(), "tokens_gerados": n,
                    "tempo_s": dt, "tok_s": tok_s})
    return out


def main():
    threads = 12
    torch.set_num_threads(threads)
    print(f"torch.set_num_threads({threads}) | threads fisicas: 6c/12t")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"carregando {MODEL_ID} (bf16, CPU)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    print("modelo carregado.")

    results = {
        "model_id": MODEL_ID,
        "threads": torch.get_num_threads(),
        "xray_params": xray_params(model),
        "xray_similaridade": xray_similarity(model),
        "codebook_k256": codebook_test(model),
        "perplexidade": perplexity(model, tok),
        "prompts": prompts_baseline(model, tok),
        "rss_pico_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nRSS pico do processo: {results['rss_pico_mb']:.0f} MB")
    print(f"JSON salvo em {OUT_JSON}")


if __name__ == "__main__":
    main()
