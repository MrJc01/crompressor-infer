#!/usr/bin/env python3
"""EXP-005 (T1): trace de localidade do roteamento MoE top-8/32 durante o decode
+ simulacao de cache LRU por camada (capacidades 4/8/16/24 experts)
+ projecao de tráfego de SSD SATA (344 MB/s medidos no EXP-001, blocos de 1 MiB).
"""
import json
from collections import OrderedDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-instruct"
OUT_JSON = "moe_cache_trace.json"
N_LAYERS, N_EXPERTS, TOP_K = 24, 32, 8
PROMPTS = [
    "Explique em 3 frases curtas como funciona uma tabela hash.",
    "What is a Mixture-of-Experts neural network? Answer concisely in 3 sentences.",
    "Escreva apenas uma função em C que soma dois vetores float de tamanho n. Sem texto extra.",
    "Resolva passo a passo e dê o resultado final: Se um disco lê a 250 MB/s e cada especialista "
    "tem 10 MB, quantos milissegundos leva para ler 8 especialistas?",
]
CAPACIDADES = [4, 8, 16, 24]
BANDA_SSD_MB_S = 344.0  # EXP-001: leitura aleatória 1 MiB, O_DIRECT
PORTES = {  # MiB por expert
    "modelo_1_3B_4bit": 0.75, "modelo_1_3B_C1_3_5bit": 0.62,
    "moe_maior_4bit": 8.0, "moe_maior_C1_3_5bit": 6.6,
}


def log(*a):
    print(*a, flush=True)


def lru_stats(steps, cap):
    """LRU por camada; processa selecoes na ordem real (camada 0..23 por token)."""
    caches = [OrderedDict() for _ in range(N_LAYERS)]
    hits = misses = 0
    miss_por_token = []
    for passo in steps:
        m_step = 0
        for l, experts in enumerate(passo):
            c = caches[l]
            for e in experts:
                if e in c:
                    c.move_to_end(e)
                    hits += 1
                else:
                    misses += 1
                    m_step += 1
                    if len(c) >= cap:
                        c.popitem(last=False)
                    c[e] = True
        miss_por_token.append(m_step)
    total = hits + misses
    return {"hits": hits, "misses": misses,
            "hit_rate_pct": 100.0 * hits / total,
            "miss_medio_por_token": sum(miss_por_token) / len(miss_por_token)}


def main():
    torch.set_num_threads(12)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True,
                                                 local_files_only=True)
    model.eval()

    trace = []  # por passo de decode: [camada][8 experts]
    def hook(mod, inp, out):
        if out.dim() == 2 and out.shape[0] == 1 and out.shape[1] == N_EXPERTS:
            trace.append([int(e) for e in out.float().topk(TOP_K, dim=-1).indices[0].tolist()])
    hooks = [model.model.layers[l].block_sparse_moe.router.layer.register_forward_hook(hook)
             for l in range(N_LAYERS)]

    traces_por_prompt = []
    with torch.no_grad():
        for i, prompt in enumerate(PROMPTS, 1):
            trace.clear()
            enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                          tokenize=False, add_generation_prompt=True)
            ids = tok(enc, return_tensors="pt")
            gen = model.generate(**ids, do_sample=False, max_new_tokens=64)
            n = gen.shape[1] - ids.input_ids.shape[1]
            steps = [trace[j] for j in range(len(trace))]
            # trace guarda 1 camada por chamada; reagrupa em passos de 24 camadas
            passos = [steps[k:k + N_LAYERS] for k in range(0, len(steps), N_LAYERS)]
            traces_por_prompt.append({"n": i, "prompt": prompt, "tokens": int(n),
                                      "passos": passos})
            log(f"prompt {i}: {n} tokens gerados, {len(passos)} passos de decode "
                f"({len(steps)} ativações de router capturadas)")

    for h in hooks:
        h.remove()

    total_sel = sum(len(p["passos"]) * N_LAYERS * TOP_K for p in traces_por_prompt)
    log(f"\nTotal de seleções capturadas no decode: {total_sel:,}")

    resultados = {}
    log(f"\n{'cap':>4} | {'hit%':>7} | {'miss/token':>10} | projeções (MB/token → tok/s @344MB/s)")
    for cap in CAPACIDADES:
        h = m_ = 0
        miss_tok = []
        for p in traces_por_prompt:
            st = lru_stats(p["passos"], cap)
            h += st["hits"]; m_ += st["misses"]; miss_tok.append(st["miss_medio_por_token"])
        miss_medio = sum(miss_tok) / len(miss_tok)
        proj = {}
        for nome, mb in PORTES.items():
            mb_tok = miss_medio * mb
            proj[nome] = {"mb_por_token": mb_tok, "toks_s_ssd": BANDA_SSD_MB_S / mb_tok}
        resultados[f"cap_{cap}"] = {"capacidade_experts": cap, "pct_ram_experts": 100.0 * cap / N_EXPERTS,
                                    "hits": h, "misses": m_, "total": h + m_,
                                    "hit_rate_pct": 100.0 * h / (h + m_),
                                    "miss_medio_por_token": miss_medio,
                                    "projecoes": proj}
        log(f"{cap:>4} | {100.0 * h / (h + m_):6.2f}% | {miss_medio:10.1f} | " +
            ", ".join(f"{k}: {v['mb_por_token']:.0f}MB→{v['toks_s_ssd']:.1f}tok/s"
                      for k, v in proj.items()))

    saida = {"model_id": MODEL_ID, "n_layers": N_LAYERS, "n_experts": N_EXPERTS,
             "top_k": TOP_K, "banda_ssd_mb_s": BANDA_SSD_MB_S, "portes": PORTES,
             "total_selecoes": total_sel, "por_capacidade": resultados,
             "traces": traces_por_prompt}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=2)
    log(f"\nJSON salvo em {OUT_JSON}")


if __name__ == "__main__":
    main()
