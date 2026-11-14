#!/usr/bin/env python3
"""EXP-004: torneio de compressao dos experts MoE (24 camadas) do
granite-3.1-1b-a400m-instruct — variantes dos Caminhos A, B e C vs Q4.

Estrategia de RAM: modelo bf16 em RAM + backup dos experts em DISCO;
por variante restaura-se do disco, quantiza tensor a tensor (aplica no
modelo), mede erro L2 e PPL. K-Means sempre com atribuicao em chunks
(||x||^2 + ||c||^2 - 2*x@c.T), chunk <= 65536.
"""
import gc
import json
import resource
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-instruct"
OUT_JSON = "torneio_results.json"
BACKUP = "experts_backup.pt"
G = 64
K1 = 256
K2 = 16
TRAIN_N = 80_000
CHUNK = 65_536
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


def log(*a):
    print(*a, flush=True)


def assign(X, C, chunk=163_840):
    """codigos de X [n,d] para centróides C [K,d] (chunks <=163840, buffer out= reutilizado)."""
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


def v_q4(W):
    Wf = W.view(-1, G).float()
    s = Wf.abs().amax(1, keepdim=True) / 7.0
    q = (Wf / s).round().clamp(-7, 7)
    bits = W.numel() * 4 + Wf.shape[0] * 16
    return (q * s), bits


def v_codebook(W, d):
    Wf, Wn, s = norm_groups(W)
    vec = Wn.reshape(-1, d)
    g = torch.Generator().manual_seed(1234)
    Xs = vec[torch.randperm(vec.shape[0], generator=g)[:TRAIN_N]]
    C = kmeans(Xs, K1, 10, 1234).half().float()  # codebook armazenado em fp16
    codes = assign(vec, C)
    Whn = C[codes.long()].view(Wf.shape)
    bits = W.numel() * 8 // d + Wf.shape[0] * 16 + K1 * d * 16
    return Whn * s, bits


def v_c1_delta(W):
    """C1: B2 (d=4,K=256) + residual esparso ternario (70% zerado)."""
    Wf, Wn, s = norm_groups(W)
    vec = Wn.reshape(-1, 4)
    g = torch.Generator().manual_seed(1234)
    Xs = vec[torch.randperm(vec.shape[0], generator=g)[:TRAIN_N]]
    C = kmeans(Xs, K1, 10, 1234).half().float()
    codes = assign(vec, C)
    Cn = C[codes.long()].view(Wf.shape)
    R = Wn - Cn
    k_keep = int(R.numel() * 0.30)
    thr = R.abs().reshape(-1).sort().values[-k_keep]
    mask = R.abs() >= thr
    st = R[mask].abs().mean().half().float()
    Whn = Cn + torch.where(mask, torch.sign(R) * st, torch.zeros(1))
    bits = (W.numel() * 2 + W.numel() * 1 + int(mask.sum()) * 1
            + Wf.shape[0] * 16 + K1 * 4 * 16 + 16)
    return Whn * s, bits


def v_c2_rvq(W):
    """C2: RVQ — B2 (d=4,K=256) + codebook 2 (K=16) no residual."""
    Wf, Wn, s = norm_groups(W)
    vec = Wn.reshape(-1, 4)
    g = torch.Generator().manual_seed(1234)
    Xs = vec[torch.randperm(vec.shape[0], generator=g)[:TRAIN_N]]
    C = kmeans(Xs, K1, 10, 1234).half().float()
    codes = assign(vec, C)
    Cn = C[codes.long()].view(Wf.shape)
    R = (Wn - Cn).reshape(-1, 4)
    Xs2 = R[torch.randperm(R.shape[0], generator=g)[:TRAIN_N]]
    C2 = kmeans(Xs2, K2, 10, 4321).half().float()
    codes2 = assign(R, C2)
    Whn = Cn + C2[codes2.long()].view(Wf.shape)
    bits = (W.numel() * 2 + W.numel() * 1 + Wf.shape[0] * 16
            + K1 * 4 * 16 + K2 * 4 * 16)
    return Whn * s, bits


def v_a_sparse50(W):
    """A: poda 50% dentro de cada grupo de 64 + Q4 nos sobreviventes."""
    Wf = W.view(-1, G).float()
    med = Wf.abs().median(1, keepdim=True).values
    mask = Wf.abs() > med
    amax = torch.where(mask, Wf.abs(), torch.zeros(1)).amax(1, keepdim=True) / 7.0
    q = (Wf / amax).round().clamp(-7, 7)
    Wh = torch.where(mask, q * amax, torch.zeros(1))
    bits = W.numel() * 1 + int(mask.sum()) * 4 + Wf.shape[0] * 16
    return Wh, bits


VARIANTS = {
    "Q4_grupo64": lambda w: v_q4(w),
    "B1_Codebook_d2_K256": lambda w: v_codebook(w, 2),
    "B2_Codebook_d4_K256": lambda w: v_codebook(w, 4),
    "C1_Hibrido_Codebook_DeltaEsparso": lambda w: v_c1_delta(w),
    "C2_Hibrido_RVQ_d4_K256_K16": lambda w: v_c2_rvq(w),
    "A_Esparso50_Q4": lambda w: v_a_sparse50(w),
}
FAMILIA_BC = {"B1_Codebook_d2_K256", "B2_Codebook_d4_K256",
              "C1_Hibrido_Codebook_DeltaEsparso", "C2_Hibrido_RVQ_d4_K256_K16"}


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


def apply_variant(model, fn):
    """Quantiza cada tensor de expert, aplica no modelo, acumula erro/bits."""
    err2 = w2 = bits = 0.0
    with torch.no_grad():
        for n, p in expert_params(model):
            Wh, b = fn(p.detach())
            Wf = p.detach().float().view(-1)
            err2 += ((Wf - Wh.reshape(-1)) ** 2).sum().item()
            w2 += (Wf * Wf).sum().item()
            bits += b
            p.data.copy_(Wh.view(p.shape).to(torch.bfloat16))
            del Wh, Wf
    return bits, (err2 / w2) ** 0.5


def compute_ppl(model, tok):
    ids = tok(VAL_TEXT, return_tensors="pt")
    with torch.no_grad():
        logits = model(**ids).logits[:, :-1].float()
    nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                            ids.input_ids[:, 1:].reshape(-1))
    return nll.item(), nll.exp().item()


def run_prompts(model, tok):
    out = []
    for i, prompt in enumerate(PROMPTS, 1):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False, max_new_tokens=128)
        dt = time.perf_counter() - t0
        n = gen.shape[1] - enc.input_ids.shape[1]
        resp = tok.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        out.append({"n": i, "prompt": prompt, "response": resp.strip(),
                    "tokens": int(n), "tempo_s": dt, "tok_s": n / dt})
        log(f"    prompt {i}: {n} tok em {dt:.1f}s ({n / dt:.2f} tok/s)")
    return out


def main():
    torch.set_num_threads(12)
    log(f"threads={torch.get_num_threads()}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True,
                                                 local_files_only=True)
    model.eval()

    log("backup dos experts em disco...")
    backup_experts(model, BACKUP)

    nll0, ppl0 = compute_ppl(model, tok)
    log(f"ref BF16: NLL={nll0:.4f} PPL={ppl0:.3f}")

    expert_bytes_bf16 = sum(p.numel() * 2 for _, p in expert_params(model))
    resultados = {}
    for nome, fn in VARIANTS.items():
        t0 = time.perf_counter()
        restore_experts(model, BACKUP)
        bits, erro = apply_variant(model, fn)
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
        log(f"{nome:34s}: {r['bits_peso']:.2f} bits/p | {r['mb_experts']:7.1f} MB | "
            f"{r['compressao_vs_bf16']:5.2f}x | err={erro:.4f} | PPL={ppl:.2f} "
            f"(Δ{r['delta_ppl_vs_bf16']:+.2f}) | {dt:.0f}s")

    abaixo = [(v, r) for v, r in resultados.items() if r["bits_peso"] < 4.5]
    ranked = sorted(abaixo, key=lambda kv: kv[1]["ppl"])
    vencedores = [ranked[0][0]]
    if vencedores[0] not in FAMILIA_BC:
        vencedores.append(next(v for v, _ in ranked if v in FAMILIA_BC))
    else:
        vencedores.append(ranked[1][0])
    log(f"\nvencedores: {vencedores}")

    textos = {}
    for v in vencedores:
        restore_experts(model, BACKUP)
        apply_variant(model, VARIANTS[v])
        log(f"\n== prompts com {v} ==")
        textos[v] = run_prompts(model, tok)

    restore_experts(model, BACKUP)
    import os
    os.remove(BACKUP)

    saida = {"model_id": MODEL_ID, "ref_bf16": {"nll": nll0, "ppl": ppl0},
             "variantes": resultados, "vencedores": vencedores,
             "prompts_vencedores": textos,
             "rss_pico_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=2)
    log(f"\nRSS pico: {saida['rss_pico_mb']:.0f} MB")
    log(f"backup removido; JSON salvo em {OUT_JSON}")


if __name__ == "__main__":
    main()
