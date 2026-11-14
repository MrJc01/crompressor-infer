#!/usr/bin/env python3
"""EXP-008: valida o motor C contra o gabarito PyTorch (golden_ref_meta.json)
e compara Modo RAM Full vs Streaming SSD (cache 768 / 576 / 384 experts).

Uso:
    python3 tools/check_exp008.py
"""
import json
import subprocess
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "engine" / "crom_cli"
GOLDEN = ROOT / "experiments" / "logs" / "baseline-hf" / "golden_ref_meta.json"
MODEL = ROOT / "artifacts" / "granite-1b.cromllm"
IDS = "37394,312,3677,1858,328,1591,15430,44"
STEPS = 32
THREADS = 6
CACHE_MODES = [768, 576, 384]

TOL_L2 = 5e-2      # tolerancia erro relativo L2 (bf16 vs fp32 + ordem de op)
TOL_FIRST6 = 6e-1  # tolerancia primeiros 6 floats (after_L23 acumula erro numerico)


def run_cli(cache, dump=True):
    cmd = [str(CLI), "--model", str(MODEL), "--ids", IDS,
           "--steps", str(STEPS), "--threads", str(THREADS),
           "--cache-experts", str(cache)]
    if dump:
        cmd.append("--dump-ckpt")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"ERRO crom_cli (cache={cache}):\n{res.stderr}")
        sys.exit(1)
    return res.stdout


def parse_ckpt(text):
    out = {}
    m = re.search(r"after_emb:\s*l2=([\d.eE+-]+)\s*first6=\[([^\]]+)\]", text)
    out["after_emb"] = (float(m.group(1)), [float(x) for x in m.group(2).split(",")])

    m = re.search(r"after_L0_attn_resid:\s*l2=([\d.eE+-]+)\s*first6=\[([^\]]+)\]", text)
    out["after_L0_attn_resid"] = (float(m.group(1)), [float(x) for x in m.group(2).split(",")])

    m = re.search(r"after_L0_moe_resid:\s*l2=([\d.eE+-]+)\s*first6=\[([^\]]+)\]", text)
    out["after_L0_moe_resid"] = (float(m.group(1)), [float(x) for x in m.group(2).split(",")])

    m = re.search(r"after_L23:\s*l2=([\d.eE+-]+)\s*first6=\[([^\]]+)\]", text)
    out["after_L23"] = (float(m.group(1)), [float(x) for x in m.group(2).split(",")])

    m = re.search(r"router L0:\s*experts=\[([^\]]+)\]\s*pesos=\[([^\]]+)\]", text)
    out["router_l0"] = {
        "experts": [int(x) for x in m.group(1).split(",")],
        "weights": [float(x) for x in m.group(2).split(",")],
    }

    m = re.search(r"router L23:\s*experts=\[([^\]]+)\]\s*pesos=\[([^\]]+)\]", text)
    out["router_l23"] = {
        "experts": [int(x) for x in m.group(1).split(",")],
        "weights": [float(x) for x in m.group(2).split(",")],
    }

    m = re.search(r"top-10 logits:(.*?)\nnext_token_id=(\d+)", text, re.S)
    parts = [p for p in re.findall(r"\[(\d+)\]=([\d.+-eE]+)", m.group(1))]
    out["top10"] = [(int(i), float(v)) for i, v in parts]
    out["next_token_id"] = int(m.group(2))

    m = re.search(r"gerados (\d+) tokens:(.*)", text)
    out["gen_tokens"] = [int(x) for x in m.group(2).split()]
    out["n_gen"] = int(m.group(1))

    m = re.search(r"load_ms=([\d.]+)\s+prefill_tok/s=([\d.]+)\s+decode_tok/s=([\d.]+)\s+RSS_MB=([\d.]+)\s+cache_hit%=([\d.]+)\s+ssd_MB=([\d.]+)\s+ssd_io_ms=([\d.]+)\s+threads=(\d+)\s+cache=(\d+)", text)
    out["bench"] = {
        "load_ms": float(m.group(1)),
        "prefill_tok_s": float(m.group(2)),
        "decode_tok_s": float(m.group(3)),
        "rss_mb": float(m.group(4)),
        "cache_hit_pct": float(m.group(5)),
        "ssd_mb": float(m.group(6)),
        "ssd_io_ms": float(m.group(7)),
        "threads": int(m.group(8)),
        "cache": int(m.group(9)),
    }
    return out


def cmp(a, b, tol, label):
    """compara (l2, first6) de a vs b."""
    err_l2 = abs(a[0] - b[0]) / (abs(b[0]) + 1e-12)
    ok = err_l2 < tol
    diffs = [abs(x - y) for x, y in zip(a[1], b[1])]
    max_d = max(diffs) if diffs else 0.0
    ok_f = max_d < tol
    status = "OK" if (ok and ok_f) else "DIVERGE"
    print(f"  {label}: l2_C={a[0]:.6f} l2_PT={b[0]:.6f} err_l2={err_l2:.2e} "
          f"max_diff_first6={max_d:.2e} -> {status}")
    return ok and ok_f


def main():
    with open(GOLDEN) as f:
        golden = json.load(f)
    gck = golden["checkpoints"]

    results = {}
    for cache in CACHE_MODES:
        print(f"\n=== Modo cache-experts={cache} ===")
        text = run_cli(cache, dump=True)
        ck = parse_ckpt(text)
        results[cache] = ck

        print("Checkpoints (C vs PyTorch golden):")
        ok_all = True
        ok_all &= cmp(ck["after_emb"], (gck["after_emb"]["l2"], gck["after_emb"]["first6"]),
                      TOL_L2, "after_emb")
        ok_all &= cmp(ck["after_L0_attn_resid"],
                      (gck["after_L0_attn_resid"]["l2"], gck["after_L0_attn_resid"]["first6"]),
                      TOL_L2, "after_L0_attn_resid")
        ok_all &= cmp(ck["after_L0_moe_resid"],
                       (gck["after_L0_moe_resid"]["l2"], gck["after_L0_moe_resid"]["first6"]),
                       TOL_L2, "after_L0_moe_resid")
        ok_all &= cmp(ck["after_L23"],
                       (gck["after_L23"]["l2"], gck["after_L23"]["first6"]),
                       6e-1, "after_L23")

        # router
        for li, key in ((0, "router_l0"), (23, "router_l23")):
            g = golden[f"routing_l{li}"]
            c_exp = ck[key]["experts"]
            g_exp = g["expert_ids"]
            c_w = ck[key]["weights"]
            g_w = g["weights"]
            exp_set_match = set(c_exp) == set(g_exp)
            w_err = max(abs(x - y) for x, y in zip(c_w, g_w)) if c_w else 1.0
            print(f"  router L{li}: experts_C={c_exp} experts_PT={g_exp} "
                  f"set_match={exp_set_match} max_w_err={w_err:.2e} -> "
                  f"{'OK' if (exp_set_match and w_err < 5e-2) else 'DIVERGE'}")
            ok_all &= exp_set_match and w_err < 5e-2

        # top-10 + next token
        g_top10 = list(zip(golden["top10_ids"], golden["top10_logits"]))
        c_top10 = ck["top10"]
        top1_match = c_top10[0][0] == g_top10[0][0]
        top3_match = [i for i, _ in c_top10[:3]] == [i for i, _ in g_top10[:3]]
        print(f"  top-10 ids C={[i for i,_ in c_top10]}")
        print(f"  top-10 ids PT={[i for i,_ in g_top10]} top3_match={top3_match}")
        print(f"  next_token_id C={ck['next_token_id']} PT={golden['next_token']['id']} "
              f"match={ck['next_token_id'] == golden['next_token']['id']}")
        ok_all &= top3_match and ck["next_token_id"] == golden["next_token"]["id"]

        print(f"  Bench: {ck['bench']}")
        results[cache]["ok"] = ok_all

    # comparacao dos tokens gerados entre os 3 modos
    print("\n=== Comparacao de tokens gerados (768 vs 576 vs 384) ===")
    gen768 = results[768]["gen_tokens"]
    gen576 = results[576]["gen_tokens"]
    gen384 = results[384]["gen_tokens"]
    print(f"  tokens 768: {gen768}")
    print(f"  tokens 576: {gen576}")
    print(f"  tokens 384: {gen384}")
    identical = (gen768 == gen576 == gen384)
    print(f"  idênticos nos 3 modos: {identical}")

    # decodificar com o tokenizer
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "ibm-granite/granite-3.1-1b-a400m-instruct", local_files_only=True)
        frase = tok.decode(gen768)
        print(f"\nTexto gerado (cache=768, {len(gen768)} tok):")
        print(f"  {frase!r}")
    except Exception as e:
        print(f"\nNao foi possivel decodificar (transformers indisponível): {e}")

    todos_ok = all(results[c].get("ok", False) for c in CACHE_MODES) and identical
    print(f"\n=== RESULTADO GERAL: {'PASS' if todos_ok else 'FALHA'} ===")
    return 0 if todos_ok else 1


if __name__ == "__main__":
    sys.exit(main())
