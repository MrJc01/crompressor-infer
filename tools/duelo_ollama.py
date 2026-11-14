#!/usr/bin/env python3
"""EXP-010: Duelo oficial crom-infer vs Ollama Q8_0 vs PyTorch BF16.

Roda o binario Go+CGO (./bin/crom-infer) nos 4 prompts padrao do projeto em dois
modos de cache (768 = RAM Full, 576 = streaming O_DIRECT 75% RAM) e compara com
os baselines ja gravados (nao chama Ollama nem sobrescreve os logs originais).
"""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin" / "crom-infer"
MODEL = ROOT / "artifacts" / "granite-1b.cromllm"
OLLAMA_BASELINE = ROOT / "experiments" / "logs" / "baseline-ollama" / "baseline_results.json"
HF_BASELINE = ROOT / "experiments" / "logs" / "baseline-hf" / "hf_baseline.json"
OUT_FILE = ROOT / "experiments" / "logs" / "crom-infer" / "duelo_ollama.json"

MAX_TOKENS = 128
THREADS = 6
MODES = [
    {"name": "crom_768", "cache_experts": 768, "timeout": 60, "label": "crom-infer RAM (768)"},
    {"name": "crom_576", "cache_experts": 576, "timeout": 180, "label": "crom-infer SSD (576)"},
]

PROMPTS = [
    "Explique em 3 frases curtas como funciona uma tabela hash.",
    "What is a Mixture-of-Experts neural network? Answer concisely in 3 sentences.",
    "Escreva apenas uma função em C que soma dois vetores float de tamanho n. Sem texto extra.",
    "Resolva passo a passo e dê o resultado final: Se um disco lê a 250 MB/s e cada especialista tem 10 MB, quantos milissegundos leva para ler 8 especialistas?",
]


def run_crom(prompt: str, cache_experts: int, timeout: int) -> dict:
    cmd = [
        str(BIN),
        "--model", str(MODEL),
        "--prompt", prompt,
        "--chat",
        "--max-tokens", str(MAX_TOKENS),
        "--threads", str(THREADS),
        "--cache-experts", str(cache_experts),
        "--json",
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise SystemExit(f"ERRO: timeout de {timeout}s em cache={cache_experts}: {prompt!r}")
    elapsed_ms = (time.time() - t0) * 1000.0

    if proc.returncode != 0:
        raise SystemExit(
            f"ERRO: crom-infer cache={cache_experts} returncode={proc.returncode}\n"
            f"stderr: {proc.stderr.strip()}"
        )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERRO: JSON invalido (cache={cache_experts}): {exc}\nstdout={proc.stdout!r}")
    ids = data.get("ids") or []
    if not ids:
        raise SystemExit(f"ERRO: ids vazio (cache={cache_experts}) para {prompt!r}")

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


def load_ollama_baseline() -> dict:
    raw = json.loads(OLLAMA_BASELINE.read_text(encoding="utf-8"))
    per_prompt = {}
    gen_tps, prompt_tps = [], []
    for r in raw.get("results", []):
        if r.get("kind") != "prompt":
            continue
        ev = r.get("eval_count", 0)
        ev_ns = r.get("eval_duration", 0)
        pe = r.get("prompt_eval_count", 0)
        pe_ns = r.get("prompt_eval_duration", 0)
        decode = (ev * 1e9 / ev_ns) if ev_ns else 0.0
        prefill = (pe * 1e9 / pe_ns) if pe_ns else 0.0
        per_prompt[r.get("n")] = {
            "prompt": r.get("prompt"),
            "text": r.get("response", ""),
            "eval_count": ev,
            "decode_tok_s": decode,
            "prefill_tok_s": prefill,
            "done_reason": r.get("full", {}).get("done_reason"),
        }
        gen_tps.append(decode)
        prompt_tps.append(prefill)

    rss_mb = None
    ps = raw.get("ollama_ps", "")
    for token in ps.split():
        if token.endswith("GB"):
            try:
                rss_mb = float(token[:-2]) * 1024.0
            except ValueError:
                pass
    return {
        "model": raw.get("model"),
        "options": raw.get("options"),
        "rss_mb": rss_mb,
        "avg_decode_tok_s": sum(gen_tps) / len(gen_tps) if gen_tps else 0.0,
        "avg_prefill_tok_s": sum(prompt_tps) / len(prompt_tps) if prompt_tps else 0.0,
        "prompts": per_prompt,
    }


def load_hf_baseline() -> dict:
    raw = json.loads(HF_BASELINE.read_text(encoding="utf-8"))
    per_prompt = {}
    tps = []
    for r in raw.get("prompts", []):
        per_prompt[r.get("n")] = {
            "prompt": r.get("prompt"),
            "text": r.get("response", ""),
            "tokens_gerados": r.get("tokens_gerados"),
            "tempo_s": r.get("tempo_s"),
            "decode_tok_s": r.get("tok_s"),
        }
        if r.get("tok_s"):
            tps.append(r["tok_s"])
    return {
        "model": raw.get("model_id"),
        "threads": raw.get("threads"),
        "rss_mb": raw.get("rss_pico_mb"),
        "perplexidade": raw.get("perplexidade"),
        "avg_decode_tok_s": sum(tps) / len(tps) if tps else 0.0,
        "prompts": per_prompt,
    }


def avg(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def main() -> int:
    if not BIN.exists():
        raise SystemExit(f"ERRO: binario nao encontrado: {BIN} (rode `make all`)")
    if not MODEL.exists():
        raise SystemExit(f"ERRO: modelo nao encontrado: {MODEL}")

    print("Carregando baselines gravados (Ollama Q8_0 + PyTorch BF16)...")
    ollama = load_ollama_baseline()
    hf = load_hf_baseline()

    results = []
    for n, prompt in enumerate(PROMPTS, 1):
        print(f"\n=== P{n}: {prompt}")
        entry = {
            "n": n,
            "prompt": prompt,
            "ollama_q8_0": ollama["prompts"].get(n),
            "pytorch_bf16": hf["prompts"].get(n),
        }
        for mode in MODES:
            print(f"  {mode['label']} ...", end="", flush=True)
            res = run_crom(prompt, mode["cache_experts"], mode["timeout"])
            entry[mode["name"]] = res
            print(f" OK ({res['elapsed_ms']:.0f} ms, {res['n_tokens']} tokens)")
        results.append(entry)

    # Confirm that streaming tokens are identical to RAM mode.
    ids_identical = all(e["crom_768"]["ids"] == e["crom_576"]["ids"] for e in results)

    summary = {}
    for mode in MODES:
        rows = [e[mode["name"]] for e in results]
        summary[mode["name"]] = {
            "label": mode["label"],
            "cache_experts": mode["cache_experts"],
            "avg_decode_tok_s": avg([r["decode_tok_s"] for r in rows]),
            "avg_prefill_tok_s": avg([r["prefill_tok_s"] for r in rows]),
            "avg_rss_mb": avg([r["rss_mb"] for r in rows]),
            "avg_cache_hit_pct": avg([r["cache_hit_pct"] for r in rows]),
            "avg_ssd_stream_mb": avg([r["ssd_stream_mb"] for r in rows]),
            "avg_attn_ms": avg([r["attn_ms"] for r in rows]),
            "avg_moe_ms": avg([r["moe_ms"] for r in rows]),
            "avg_lm_head_ms": avg([r["lm_head_ms"] for r in rows]),
        }

    payload = {
        "experiment": "EXP-010",
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "config": {
            "binary": str(BIN.relative_to(ROOT)),
            "model": str(MODEL.relative_to(ROOT)),
            "threads": THREADS,
            "max_tokens": MAX_TOKENS,
            "chat_template": True,
            "modes": MODES,
        },
        "baselines": {
            "ollama_q8_0": {
                "model": ollama["model"],
                "options": ollama["options"],
                "avg_decode_tok_s": ollama["avg_decode_tok_s"],
                "avg_prefill_tok_s": ollama["avg_prefill_tok_s"],
                "rss_mb": ollama["rss_mb"],
            },
            "pytorch_bf16": {
                "model": hf["model"],
                "threads": hf["threads"],
                "avg_decode_tok_s": hf["avg_decode_tok_s"],
                "rss_mb": hf["rss_mb"],
                "perplexidade": hf["perplexidade"],
            },
        },
        "summary": summary,
        "crom_768_vs_576_ids_identical": ids_identical,
        "prompts": results,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ---- Resumo tabular ----
    print("\n\n" + "=" * 106)
    print("DUELO EXP-010 — 4 prompts padrao, chat template, greedy (temp=0)")
    print("=" * 106)
    header = f"{'Engine':<26}{'decode tok/s':>14}{'prefill tok/s':>15}{'RSS MB':>10}{'cache_hit %':>13}{'ssd_stream MB':>15}"
    print(header)
    print("-" * len(header))
    print(f"{'Ollama Q8_0 (baseline)':<26}{ollama['avg_decode_tok_s']:>14.2f}{ollama['avg_prefill_tok_s']:>15.2f}{('%.1f' % ollama['rss_mb']) if ollama['rss_mb'] else 'n/a':>10}{'n/a':>13}{'n/a':>15}")
    print(f"{'PyTorch BF16 (baseline)':<26}{hf['avg_decode_tok_s']:>14.2f}{'n/a':>15}{('%.1f' % hf['rss_mb']) if hf['rss_mb'] else 'n/a':>10}{'n/a':>13}{'n/a':>15}")
    for mode in MODES:
        s = summary[mode["name"]]
        print(f"{mode['label']:<26}{s['avg_decode_tok_s']:>14.2f}{s['avg_prefill_tok_s']:>15.2f}{s['avg_rss_mb']:>10.1f}{s['avg_cache_hit_pct']:>13.2f}{s['avg_ssd_stream_mb']:>15.2f}")
    print("-" * len(header))
    print(f"ids 768 == ids 576 nos 4 prompts: {ids_identical}")

    # ---- Respostas literais (768) ----
    print("\n" + "=" * 106)
    print("RESPOSTAS LITERAIS — crom-infer RAM (768)")
    print("=" * 106)
    for e in results:
        r = e["crom_768"]
        print(f"\n--- P{e['n']}: {e['prompt']}")
        print(f"[decode={r['decode_tok_s']:.2f} tok/s, n_tokens={r['n_tokens']}, "
              f"ids_576_identicos={e['crom_768']['ids'] == e['crom_576']['ids']}]")
        print(r["text"])
    print(f"\nResultado salvo em: {OUT_FILE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
