#!/usr/bin/env python3
"""EXP-006: qualidade (PPL) abaixo de 4 bits/peso nos experts do
granite-3.1-1b-a400m-instruct — Q3, LUT16 K=16 (d=1/d=2), C3 (d2 + delta
1-bit denso), C1_v2 (d4 K256 + ternario por grupo) e Hibrido_Misto
(Q4 nas camadas extremas + melhor 3.x-bit nas centrais).

Mesma estrategia do EXP-004: backup dos experts em disco, restaura ->
quantiza tensor a tensor -> erro L2 + PPL no VAL_TEXT. Sem prompts.
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
OUT_JSON = "qualidade_v2.json"
BACKUP = "experts_backup.pt"
G = 64
K16 = 16
K256 = 256
TRAIN_N = 30_000
SAMPLE_Q = 200_000
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


def assign(X, C, chunk=163_840):
    """codigos de X [n,d] para centroides C [K,d] (chunks, buffer out= reutilizado)."""
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
    """W bf16 -> grupos de 64 normalizados por max_abs (escala fp16)."""
    Wf = W.view(-1, G).float()
    s = Wf.abs().amax(1, keepdim=True).half().float().clamp(min=1e-12)
    return Wf, Wf / s, s


def v_q3(W):
    """Q3: int3 simetrico [-3..3], escala fp16 por grupo de 64 (3.25 bits)."""
    Wf = W.view(-1, G).float()
    s = (Wf.abs().amax(1, keepdim=True) / 3.0).half().float().clamp(min=1e-12)
    q = (Wf / s).round().clamp(-3, 3)
    bits = W.numel() * 3 + Wf.shape[0] * 16
    return q * s, bits


def v_q4(W):
    """Q4: int4 simetrico [-7..7], escala fp16 por grupo de 64 (4.25 bits)."""
    Wf = W.view(-1, G).float()
    s = (Wf.abs().amax(1, keepdim=True) / 7.0).half().float().clamp(min=1e-12)
    q = (Wf / s).round().clamp(-7, 7)
    bits = W.numel() * 4 + Wf.shape[0] * 16
    return q * s, bits


def _lut16(W, d):
    """base comum: K-Means K=16 no dominio normalizado (codebook por tensor)."""
    Wf, Wn, s = norm_groups(W)
    vec = Wn.reshape(-1, d)
    g = torch.Generator().manual_seed(1234)
    Xs = vec[torch.randperm(vec.shape[0], generator=g)[:TRAIN_N]]
    C = kmeans(Xs, K16, 8, 1234).half().float()
    codes = assign(vec, C)
    Cn = C[codes.long()].view(Wf.shape)
    return Wf, Wn, s, Cn


def v_lut16_d1(W):
    """LUT16 d=1 K=16: 4 bits/peso, codebook compartilhado por tensor (4.25)."""
    Wf, Wn, s, Cn = _lut16(W, 1)
    bits = W.numel() * 4 + Wf.shape[0] * 16 + K16 * 1 * 16
    return Cn * s, bits


def v_lut16_d2(W):
    """LUT16 d=2 K=16: 2 bits/peso de codigo + escalas (2.25)."""
    Wf, Wn, s, Cn = _lut16(W, 2)
    bits = W.numel() * 2 + Wf.shape[0] * 16 + K16 * 2 * 16
    return Cn * s, bits


def v_c3(W):
    """C3: LUT16 d=2 K=16 (2.25) + delta denso 1-bit (sign(R)*alpha por
    grupo de 64, sem bitmap e sem saltos) -> 3.50 bits."""
    Wf, Wn, s, Cn = _lut16(W, 2)
    R = Wn - Cn
    alpha = R.abs().mean(1, keepdim=True).half().float()
    Wh = Cn + torch.sign(R) * alpha
    bits = W.numel() * 2 + W.numel() * 1 + Wf.shape[0] * 16 * 2 + K16 * 2 * 16
    return Wh * s, bits


def v_c1_v2(W):
    """C1_v2: codebook d=4 K=256 (TRAIN_N=30k, 8 iters) + residual: 70%
    menores erros zerados (torch.quantile em amostra de 200k), 30%
    ternarios {-1,0,+1} com escala fp16 POR GRUPO de 64."""
    Wf, Wn, s = norm_groups(W)
    vec = Wn.reshape(-1, 4)
    g = torch.Generator().manual_seed(1234)
    Xs = vec[torch.randperm(vec.shape[0], generator=g)[:TRAIN_N]]
    C = kmeans(Xs, K256, 8, 1234).half().float()
    codes = assign(vec, C)
    Cn = C[codes.long()].view(Wf.shape)
    R = Wn - Cn
    flat = R.abs().reshape(-1)
    idx = torch.randperm(flat.numel(), generator=g)[:SAMPLE_Q]
    thr = torch.quantile(flat[idx], 0.70).item()
    mask = R.abs() >= thr
    kept = int(mask.sum())
    alpha = ((R.abs() * mask).sum(1, keepdim=True)
             / mask.sum(1, keepdim=True).clamp(min=1)).half().float()
    Wh = Cn + torch.where(mask, torch.sign(R) * alpha, torch.zeros(1))
    bits = (W.numel() * 2 + kept * 2 + Wf.shape[0] * 16 * 2
            + K256 * 4 * 16 + 16)
    return Wh * s, bits


VARIANTS = {
    "Q3_grupo64": v_q3,
    "LUT16_d1_K16": v_lut16_d1,
    "LUT16_d2_K16": v_lut16_d2,
    "C3_LUT16_d2_plus_1bit": v_c3,
    "C1_v2_K256_DeltaGrupo": v_c1_v2,
}


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


def _aplica_tensor(p, fn, acc):
    Wh, b = fn(p.detach())
    Wf = p.detach().float().view(-1)
    acc[0] += ((Wf - Wh.reshape(-1)) ** 2).sum().item()
    acc[1] += (Wf * Wf).sum().item()
    acc[2] += b
    p.data.copy_(Wh.view(p.shape).to(torch.bfloat16))
    del Wh, Wf


def apply_fn(model, fn):
    """quantiza cada tensor de expert com fn, aplica no modelo, da erro/bits."""
    acc = [0.0, 0.0, 0.0]
    with torch.no_grad():
        for n, p in expert_params(model):
            _aplica_tensor(p, fn, acc)
    return acc[2], (acc[0] / acc[1]) ** 0.5


def apply_mixed(model, fn_extrema, fn_central):
    """Hibrido_Misto: fn_extrema nas 8 camadas extremas, fn_central nas 16 centrais."""
    acc = [0.0, 0.0, 0.0]
    with torch.no_grad():
        for n, p in expert_params(model):
            layer = int(re.search(r"layers\.(\d+)\.", n).group(1))
            _aplica_tensor(p, fn_extrema if layer in EXTREMAS else fn_central, acc)
    return acc[2], (acc[0] / acc[1]) ** 0.5


def compute_ppl(model, tok):
    ids = tok(VAL_TEXT, return_tensors="pt")
    with torch.no_grad():
        logits = model(**ids).logits[:, :-1].float()
    nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                            ids.input_ids[:, 1:].reshape(-1))
    return nll.item(), nll.exp().item()


def main():
    torch.set_num_threads(6)
    log(f"threads={torch.get_num_threads()}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True,
                                                 local_files_only=True)
    model.eval()

    log("backup dos experts em disco...")
    backup_experts(model, BACKUP)
    try:
        nll0, ppl0 = compute_ppl(model, tok)
        log(f"ref BF16: NLL={nll0:.4f} PPL={ppl0:.3f}")
        expert_bytes_bf16 = sum(p.numel() * 2 for _, p in expert_params(model))

        resultados = {}
        for nome, fn in VARIANTS.items():
            t0 = time.perf_counter()
            restore_experts(model, BACKUP)
            bits, erro = apply_fn(model, fn)
            nll, ppl = compute_ppl(model, tok)
            dt = time.perf_counter() - t0
            resultados[nome] = {
                "bits_peso": bits / 1_207_959_552,
                "mb_experts": bits / 8 / 2**20,
                "compressao_vs_bf16": expert_bytes_bf16 / (bits / 8),
                "erro_l2": erro, "nll": nll, "ppl": ppl,
                "delta_ppl_vs_bf16": ppl - ppl0, "tempo_s": dt,
            }
            r = resultados[nome]
            log(f"{nome:24s}: {r['bits_peso']:.2f} bits/p | {r['mb_experts']:7.1f} MB | "
                f"{r['compressao_vs_bf16']:5.2f}x | err={erro:.4f} | PPL={ppl:.2f} "
                f"(Δ{r['delta_ppl_vs_bf16']:+.2f}) | {dt:.0f}s")

        melhor_central = ("C3_LUT16_d2_plus_1bit"
                          if resultados["C3_LUT16_d2_plus_1bit"]["ppl"]
                          <= resultados["Q3_grupo64"]["ppl"] else "Q3_grupo64")
        log(f"\nHibrido_Misto: extremas (0-3,20-23) = Q4_grupo64; "
            f"centrais (4..19) = {melhor_central}")
        t0 = time.perf_counter()
        restore_experts(model, BACKUP)
        bits, erro = apply_mixed(model, v_q4, VARIANTS[melhor_central])
        nll, ppl = compute_ppl(model, tok)
        dt = time.perf_counter() - t0
        resultados["Hibrido_Misto"] = {
            "bits_peso": bits / 1_207_959_552,
            "mb_experts": bits / 8 / 2**20,
            "compressao_vs_bf16": expert_bytes_bf16 / (bits / 8),
            "erro_l2": erro, "nll": nll, "ppl": ppl,
            "delta_ppl_vs_bf16": ppl - ppl0, "tempo_s": dt,
            "central_escolhida": melhor_central,
        }
        r = resultados["Hibrido_Misto"]
        log(f"{'Hibrido_Misto':24s}: {r['bits_peso']:.2f} bits/p | {r['mb_experts']:7.1f} MB | "
            f"{r['compressao_vs_bf16']:5.2f}x | err={erro:.4f} | PPL={ppl:.2f} "
            f"(Δ{r['delta_ppl_vs_bf16']:+.2f}) | {dt:.0f}s")
    finally:
        restore_experts(model, BACKUP)
        if os.path.exists(BACKUP):
            os.remove(BACKUP)
        log("backup removido")

    saida = {"model_id": MODEL_ID,
             "ref_bf16": {"nll": nll0, "ppl": ppl0},
             "variantes": resultados,
             "rss_pico_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=2)
    log(f"RSS pico: {saida['rss_pico_mb']:.0f} MB")
    log(f"JSON salvo em {OUT_JSON}")


if __name__ == "__main__":
    main()
