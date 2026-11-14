#!/usr/bin/env python3
"""EXP-013A: Benchmark rapido de qualidade C1_TOPK (Codebook d=4, K=256 + Top-K deltas
esparsos por grupo de 64) com seguranca estrita de memoria (< 4.5 GiB RSS).

Testa 5 perfis no mesmo VAL_TEXT:
1. C1_topk4: 4 maiores |R| por grupo de 64 (3.00 bits/peso)
2. C1_topk8: 8 maiores |R| por grupo de 64 (3.50 bits/peso)
3. C1_topk12: 12 maiores |R| por grupo de 64 (4.00 bits/peso)
4. Hibrido_Q4_C1_topk8: camadas extremas (0..3, 20..23) em Q4_g64 e centrais (4..19) em C1_topk8
5. Hibrido_Q4_C1_topk4: camadas extremas (0..3, 20..23) em Q4_g64 e centrais (4..19) em C1_topk4
"""
import gc
import json
import os
import re
import resource
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-instruct"
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c1_topk_results.json")
BACKUP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experts_backup.pt")
G = 64
K256 = 256
TRAIN_N = 30_000
EXTREMAS = {0, 1, 2, 3, 20, 21, 22, 23}
VAL_TEXT = (
    "A Mixture-of-Experts model routes each token to a few experts, which is why reading the "
    "right experts from disk matters more than loading everything. Uma tabela hash mapeia chaves "
    "para baldes usando uma função de hash, e colisões são resolvidas por encadeamento ou "
    "endereçamento aberto. Sparse deltas between experts can be stored on a SATA SSD and pulled "
    "on demand, since reads of 1 MiB saturate the bandwidth. O codebook transforma multiplicação "
    "em busca de tabela na cache L1/L2 do processador, evitando FLOPs no caminho crítico."
)

def log(*a):
    print(*a, flush=True)

def get_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def assign(X, C, chunk=163_840):
    cn = (C * C).sum(1)
    xn = (X * X).sum(1)
    codes = torch.empty(X.shape[0], dtype=torch.int32)
    buf = torch.empty(min(chunk, X.shape[0]), C.shape[0])
    for i in range(0, X.shape[0], chunk):
        xb = X[i:i + chunk]
        b = buf[:xb.shape[0]]
        torch.mm(xb, C.T, out=b)
        b.mul_(-2.0).add_(xn[i:i + chunk, None]).add_(cn)
        codes[i:i + chunk] = b.argmin(1)
    return codes

def kmeans(X, k, iters, seed):
    g = torch.Generator().manual_seed(seed)
    C = X[torch.randperm(X.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        codes = assign(X, C).long()
        S = torch.zeros(k, X.shape[1])
        cnt = torch.zeros(k)
        S.index_add_(0, codes, X)
        cnt.index_add_(0, codes, torch.ones(X.shape[0]))
        nz = cnt > 0
        C[nz] = S[nz] / cnt[nz, None]
    return C

def norm_groups(W):
    Wf = W.view(-1, G).float()
    s = Wf.abs().amax(1, keepdim=True).half().float().clamp(min=1e-12)
    return Wf, Wf / s, s

def quant_q4_tensor(W):
    Wf = W.view(-1, G).float()
    s = (Wf.abs().amax(1, keepdim=True) / 7.0).half().float().clamp(min=1e-12)
    q = (Wf / s).round().clamp(-7, 7)
    bits = W.numel() * 4 + Wf.shape[0] * 16
    Wh = q * s
    return Wh.view(W.shape).to(torch.bfloat16), bits

def quant_c1_tensor(W, C, k_delta):
    Wf, Wn, s_base = norm_groups(W)
    vec = Wn.reshape(-1, 4)
    codes = assign(vec, C)
    Cn = C[codes.long()].view(Wf.shape)
    R = Wn - Cn

    top_vals, top_idx = torch.topk(R.abs(), k=k_delta, dim=-1)
    alpha = top_vals.mean(dim=1, keepdim=True).half().float()

    delta = torch.zeros_like(Wf)
    signs = torch.gather(R, 1, top_idx).sign()
    delta.scatter_(1, top_idx, signs * alpha)

    Wh = (Cn + delta) * s_base
    cb_bits = K256 * 4 * 32
    codes_bits = W.numel() * 2
    num_groups = Wf.shape[0]
    bits = cb_bits + codes_bits + num_groups * (k_delta * 8 + 32)
    return Wh.view(W.shape).to(torch.bfloat16), bits

def expert_params(model):
    for n, p in model.named_parameters():
        if "block_sparse_moe" in n and ("input_linear" in n or "output_linear" in n):
            yield n, p

def backup_experts(model, path):
    torch.save({n: p.detach() for n, p in expert_params(model)}, path)

def restore_experts(model, path):
    bk = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    for n, p in expert_params(model):
        p.data.copy_(bk[n])
    del bk
    gc.collect()

def compute_ppl(model, tok):
    ids = tok(VAL_TEXT, return_tensors="pt")
    with torch.no_grad():
        logits = model(**ids).logits[:, :-1].float()
    nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                            ids.input_ids[:, 1:].reshape(-1))
    return nll.item(), nll.exp().item()

def main():
    torch.set_num_threads(6)
    log(f"Carregando {MODEL_ID} com {torch.get_num_threads()} threads... (RSS={get_rss_mb():.1f} MB)")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True,
                                                 local_files_only=True)
    model.eval()
    log(f"Modelo carregado. RSS={get_rss_mb():.1f} MB")

    log("Criando backup dos experts em disco...")
    backup_experts(model, BACKUP)
    try:
        nll0, ppl0 = compute_ppl(model, tok)
        log(f"Referência BF16: NLL={nll0:.4f} PPL={ppl0:.3f}")
        total_expert_elems = sum(p.numel() for _, p in expert_params(model))
        expert_bytes_bf16 = total_expert_elems * 2

        # Treina 1 único Codebook d=4 K=256 para cada uma das 48 matrizes de experts
        # Total de RAM para 48 codebooks: 48 * 256 * 4 * 4 bytes = 196.608 bytes (~192 KB!)
        log("Treinando Codebooks d=4 K=256 para as 48 matrizes de experts...")
        codebooks = {}
        t_cb0 = time.perf_counter()
        with torch.no_grad():
            for idx_p, (n, p) in enumerate(expert_params(model)):
                W = p.detach()
                Wf, Wn, _ = norm_groups(W)
                vec = Wn.reshape(-1, 4)
                g = torch.Generator().manual_seed(1234)
                Xs = vec[torch.randperm(vec.shape[0], generator=g)[:TRAIN_N]]
                C = kmeans(Xs, K256, 8, 1234).half().float()
                codebooks[n] = C
                del Wf, Wn, vec, Xs
                if (idx_p + 1) % 12 == 0:
                    log(f"  treinados {idx_p + 1}/48 codebooks... (RSS={get_rss_mb():.1f} MB)")
        gc.collect()
        log(f"Codebooks treinados em {time.perf_counter() - t_cb0:.1f}s. RSS={get_rss_mb():.1f} MB")

        # Definição dos perfis a serem testados
        perfis = [
            ("Q4_g64_ref", lambda n, layer, p: quant_q4_tensor(p)),
            ("C1_topk4", lambda n, layer, p: quant_c1_tensor(p, codebooks[n], 4)),
            ("C1_topk8", lambda n, layer, p: quant_c1_tensor(p, codebooks[n], 8)),
            ("C1_topk12", lambda n, layer, p: quant_c1_tensor(p, codebooks[n], 12)),
            ("Hibrido_Q4_C1_topk8", lambda n, layer, p: quant_q4_tensor(p) if layer in EXTREMAS else quant_c1_tensor(p, codebooks[n], 8)),
            ("Hibrido_Q4_C1_topk4", lambda n, layer, p: quant_q4_tensor(p) if layer in EXTREMAS else quant_c1_tensor(p, codebooks[n], 4)),
        ]

        resultados = {
            "BF16_ref": {
                "bits_peso": 16.0,
                "mb_experts": round(expert_bytes_bf16 / 2**20, 1),
                "l2_erro_rel": 0.0,
                "nll": round(nll0, 4),
                "ppl": round(ppl0, 3),
                "delta_ppl": 0.0,
                "rss_pico_mb": round(get_rss_mb(), 1),
            }
        }

        log("\n" + "=" * 102)
        log(f"{'Perfil':<22} | {'Bits/peso':<9} | {'MB Exp':<8} | {'Erro L2':<8} | {'NLL':<7} | {'PPL':<7} | {'Δ PPL':<7} | {'RSS Pico':<8}")
        log("-" * 102)
        log(f"{'BF16_ref':<22} | {'16.00':<9} | {expert_bytes_bf16 / 2**20:<8.1f} | {'0.0000':<8} | {nll0:<7.4f} | {ppl0:<7.3f} | {'+0.00':<7} | {get_rss_mb():<8.1f}")

        for nome, quant_fn in perfis:
            t0 = time.perf_counter()
            restore_experts(model, BACKUP)

            tot_bits = 0
            err_num = 0.0
            err_den = 0.0

            with torch.no_grad():
                for n, p in expert_params(model):
                    layer = int(re.search(r"layers\.(\d+)\.", n).group(1))
                    W_orig = p.detach().clone().float().view(-1)
                    Wh_bf16, bits = quant_fn(n, layer, p.detach())
                    tot_bits += bits
                    Wh_float = Wh_bf16.float().view(-1)
                    err_num += ((W_orig - Wh_float) ** 2).sum().item()
                    err_den += (W_orig * W_orig).sum().item()
                    p.data.copy_(Wh_bf16)
                    del W_orig, Wh_bf16, Wh_float

            gc.collect()
            l2_rel = (err_num / err_den) ** 0.5
            nll, ppl = compute_ppl(model, tok)
            dt = time.perf_counter() - t0
            bits_peso = tot_bits / total_expert_elems
            mb_exp = tot_bits / 8 / 2**20
            delta_ppl = ppl - ppl0
            rss_pico = get_rss_mb()

            resultados[nome] = {
                "bits_peso": round(bits_peso, 4),
                "mb_experts": round(mb_exp, 2),
                "l2_erro_rel": round(l2_rel, 5),
                "nll": round(nll, 4),
                "ppl": round(ppl, 3),
                "delta_ppl": round(delta_ppl, 3),
                "rss_pico_mb": round(rss_pico, 1),
                "tempo_s": round(dt, 2),
            }

            log(f"{nome:<22} | {bits_peso:<9.2f} | {mb_exp:<8.1f} | {l2_rel:<8.4f} | {nll:<7.4f} | {ppl:<7.3f} | {delta_ppl:+7.2f} | {rss_pico:<8.1f}")

        log("=" * 102 + "\n")

        with open(OUT_JSON, "w") as f:
            json.dump(resultados, f, indent=2)
        log(f"Resultados gravados em {OUT_JSON}")

    finally:
        if os.path.exists(BACKUP):
            log(f"Removendo arquivo de backup temporario {BACKUP}...")
            os.remove(BACKUP)
        gc.collect()

if __name__ == "__main__":
    main()
