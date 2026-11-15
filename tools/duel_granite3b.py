#!/usr/bin/env python3
"""EXP-015: Duelo oficial MoE 3.3B (Granite-3.1-3B MoE).
Compara:
- Ollama (granite3.1-moe:3b, Q4_K_M)
- crom-infer Q4 (1280 experts = 100% RAM)
- crom-infer Q4 (640 experts = 50% RAM)
- crom-infer Híbrido C1_K8 (1280 experts = 100% RAM)
- crom-infer Híbrido C1_K8 (640 experts = 50% RAM)

Salva em experiments/logs/crom-infer/exp015_granite3b_duel.json.
"""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin" / "crom-infer"
MODEL_Q4 = ROOT / "artifacts" / "granite-3b-q4.cromllm"
MODEL_HYBRID = ROOT / "artifacts" / "granite-3b-hybrid.cromllm"
OUT_FILE = ROOT / "experiments" / "logs" / "crom-infer" / "exp015_granite3b_duel.json"

MAX_TOKENS = 128
THREADS = 6
REPEAT_PENALTY = 1.15

PROMPTS = [
    "Explique em 3 frases curtas como funciona uma tabela hash.",
    "What is a Mixture-of-Experts neural network? Answer concisely in 3 sentences.",
    "Escreva apenas uma função em C que soma dois vetores float de tamanho n. Sem texto extra.",
    "Resolva passo a passo e dê o resultado final: Se um disco lê a 250 MB/s e cada especialista tem 10 MB, quantos milissegundos leva para ler 8 especialistas?",
]


def run_crom(model_path: Path, prompt: str, cache_experts: int, timeout: int = 120) -> dict:
    cmd = [
        str(BIN),
        "--model", str(model_path),
        "--prompt", prompt,
        "--chat",
        "--max-tokens", str(MAX_TOKENS),
        "--threads", str(THREADS),
        "--cache-experts", str(cache_experts),
        "--repeat-penalty", str(REPEAT_PENALTY),
        "--json",
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise SystemExit(f"ERRO: timeout de {timeout}s em model={model_path.name} cache={cache_experts}: {prompt!r}")
    elapsed_ms = (time.time() - t0) * 1000.0

    if proc.returncode != 0:
        raise SystemExit(
            f"ERRO: crom-infer returncode={proc.returncode}\n"
            f"model={model_path.name} cache={cache_experts}\n"
            f"stderr: {proc.stderr.strip()}"
        )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERRO: JSON invalido: {exc}\nstdout={proc.stdout!r}")
    ids = data.get("ids") or []
    if not ids:
        raise SystemExit(f"ERRO: ids vazio para {prompt!r}")

    return {
        "text": data.get("text", ""),
        "ids": ids,
        "n_tokens": len(ids),
        "load_ms": data.get("load_ms"),
        "prefill_tok_s": data.get("prefill_tok_s"),
        "decode_tok_s": data.get("decode_tok_s"),
        "rss_mb": data.get("rss_mb"),
        "cache_hit_pct": data.get("cache_hit_pct"),
        "ssd_stream_mb": data.get("ssd_stream_mb"),
        "attn_ms": data.get("attn_ms"),
        "moe_ms": data.get("moe_ms"),
        "lm_head_ms": data.get("lm_head_ms"),
        "elapsed_ms": elapsed_ms,
    }


def run_ollama(prompt: str, timeout: int = 120) -> dict:
    data = json.dumps({
        "model": "granite3.1-moe:3b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": MAX_TOKENS,
            "num_thread": THREADS,
            "temperature": 0.0,
        }
    }).encode("utf-8")

    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            res = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise SystemExit(f"ERRO Ollama: {exc}")
    elapsed_ms = (time.time() - t0) * 1000.0

    eval_count = res.get("eval_count", 0)
    eval_dur = res.get("eval_duration", 1)
    prompt_count = res.get("prompt_eval_count", 0)
    prompt_dur = res.get("prompt_eval_duration", 1)

    decode_tok_s = eval_count / (eval_dur / 1e9) if eval_dur > 0 else 0.0
    prefill_tok_s = prompt_count / (prompt_dur / 1e9) if prompt_dur > 0 else 0.0

    rss_mb = 0.0
    try:
        ps = subprocess.run(["ps", "-C", "llama-server", "-o", "rss="], capture_output=True, text=True)
        if ps.returncode == 0 and ps.stdout.strip():
            rss_mb = sum(int(x) for x in ps.stdout.split()) / 1024.0
    except Exception:
        pass

    return {
        "text": res.get("response", ""),
        "n_tokens": eval_count,
        "load_ms": res.get("load_duration", 0) / 1e6,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
        "rss_mb": round(rss_mb, 1),
        "elapsed_ms": elapsed_ms,
    }


def main():
    print(f"=== DUELO OFICIAL: GRANITE-3.1-3B (3.3B MoE) EXP-015 ===")
    print(f"Model Q4:     {MODEL_Q4} ({MODEL_Q4.stat().st_size / 1e6:.1f} MB)")
    print(f"Model Hybrid: {MODEL_HYBRID} ({MODEL_HYBRID.stat().st_size / 1e6:.1f} MB)")
    print(f"Threads: {THREADS} | Max Tokens: {MAX_TOKENS} | Repetition Penalty: {REPEAT_PENALTY}\n")

    results = {
        "timestamp": _dt.datetime.now().isoformat(),
        "config": {
            "max_tokens": MAX_TOKENS,
            "threads": THREADS,
            "repeat_penalty": REPEAT_PENALTY,
            "q4_model_size_bytes": MODEL_Q4.stat().st_size,
            "hybrid_model_size_bytes": MODEL_HYBRID.stat().st_size,
        },
        "ollama": {},
        "crom_modes": {},
    }

    # 1. Ollama Baseline
    print("-------------------------------------------------------------")
    print(">>> EXECUTANDO OLLAMA (granite3.1-moe:3b, Q4_K_M)")
    print("-------------------------------------------------------------")
    ollama_runs = []
    for p_idx, p in enumerate(PROMPTS):
        print(f"  Prompt {p_idx+1}/4: {p[:45]}...", end="", flush=True)
        res = run_ollama(p, timeout=120)
        print(f" -> {res['decode_tok_s']:.2f} tok/s ({res['n_tokens']} tok, RSS={res['rss_mb']} MB)")
        ollama_runs.append({
            "prompt_idx": p_idx + 1,
            "prompt": p,
            "metrics": res,
        })
    avg_prefill = sum(x["metrics"]["prefill_tok_s"] for x in ollama_runs) / len(ollama_runs)
    avg_decode = sum(x["metrics"]["decode_tok_s"] for x in ollama_runs) / len(ollama_runs)
    max_rss = max(x["metrics"]["rss_mb"] for x in ollama_runs)
    results["ollama"] = {
        "summary": {
            "avg_prefill_tok_s": round(avg_prefill, 2),
            "avg_decode_tok_s": round(avg_decode, 2),
            "max_rss_mb": round(max_rss, 1),
        },
        "prompts": ollama_runs,
    }
    print(f"  [RESUMO OLLAMA] decode={avg_decode:.2f} t/s, prefill={avg_prefill:.2f} t/s, RSS={max_rss:.1f} MB\n")

    # 2. crom-infer modes
    cache_levels = [1280, 640]
    models = [
        ("q4", MODEL_Q4),
        ("hybrid", MODEL_HYBRID),
    ]

    for cache in cache_levels:
        c_key = str(cache)
        results["crom_modes"][c_key] = {}
        print("-------------------------------------------------------------")
        pct_label = "100% RAM" if cache == 1280 else "50% RAM"
        print(f">>> TESTANDO CROM-INFER: CACHE = {cache} EXPERTS ({pct_label})")
        print("-------------------------------------------------------------")

        for m_key, m_path in models:
            print(f"  Modelo: {m_key.upper()} ({m_path.name}) ...")
            prompt_res = []
            for p_idx, p in enumerate(PROMPTS):
                print(f"    Prompt {p_idx+1}/4: {p[:45]}...", end="", flush=True)
                t_run = run_crom(m_path, p, cache, timeout=120)
                print(f" -> {t_run['decode_tok_s']:.2f} tok/s ({t_run['n_tokens']} tok, stream={t_run['ssd_stream_mb']:.1f} MB)")
                prompt_res.append({
                    "prompt_idx": p_idx + 1,
                    "prompt": p,
                    "metrics": t_run,
                })

            avg_prefill = sum(x["metrics"]["prefill_tok_s"] for x in prompt_res) / len(prompt_res)
            avg_decode = sum(x["metrics"]["decode_tok_s"] for x in prompt_res) / len(prompt_res)
            avg_hit = sum(x["metrics"]["cache_hit_pct"] for x in prompt_res) / len(prompt_res)
            total_stream = sum(x["metrics"]["ssd_stream_mb"] for x in prompt_res)
            max_rss = max(x["metrics"]["rss_mb"] for x in prompt_res)

            results["crom_modes"][c_key][m_key] = {
                "summary": {
                    "avg_prefill_tok_s": round(avg_prefill, 2),
                    "avg_decode_tok_s": round(avg_decode, 2),
                    "avg_cache_hit_pct": round(avg_hit, 2),
                    "total_ssd_stream_mb": round(total_stream, 2),
                    "max_rss_mb": round(max_rss, 1),
                },
                "prompts": prompt_res,
            }

        q4_sum = results["crom_modes"][c_key]["q4"]["summary"]
        hy_sum = results["crom_modes"][c_key]["hybrid"]["summary"]
        stream_saved = q4_sum["total_ssd_stream_mb"] - hy_sum["total_ssd_stream_mb"]
        stream_pct = (stream_saved / q4_sum["total_ssd_stream_mb"] * 100) if q4_sum["total_ssd_stream_mb"] > 0 else 0.0

        print(f"\n  [RESUMO CACHE={cache} ({pct_label})]")
        print(f"  Q4:     decode={q4_sum['avg_decode_tok_s']} t/s, prefill={q4_sum['avg_prefill_tok_s']} t/s, stream={q4_sum['total_ssd_stream_mb']} MB, RSS={q4_sum['max_rss_mb']} MB")
        print(f"  HYBRID: decode={hy_sum['avg_decode_tok_s']} t/s, prefill={hy_sum['avg_prefill_tok_s']} t/s, stream={hy_sum['total_ssd_stream_mb']} MB, RSS={hy_sum['max_rss_mb']} MB")
        if stream_saved > 0:
            print(f"  Economia de I/O SSD no Híbrido: -{stream_saved:.1f} MB (-{stream_pct:.1f}%)")
        print()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Resultados completos salvos em: {OUT_FILE}")


if __name__ == "__main__":
    main()
