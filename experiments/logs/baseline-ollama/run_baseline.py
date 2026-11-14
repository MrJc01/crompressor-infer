#!/usr/bin/env python3
"""EXP-002: baseline oficial do granite3.1-moe:1b no Ollama (API /api/generate)."""
import json
import subprocess
import urllib.request

URL = "http://localhost:11434/api/generate"
MODEL = "granite3.1-moe:1b"
OPTS = {"temperature": 0.0, "seed": 42, "num_predict": 128, "num_gpu": 0}
PROMPTS = [
    "Explique em 3 frases curtas como funciona uma tabela hash.",
    "What is a Mixture-of-Experts neural network? Answer concisely in 3 sentences.",
    "Escreva apenas uma função em C que soma dois vetores float de tamanho n. Sem texto extra.",
    "Resolva passo a passo e dê o resultado final: Se um disco lê a 250 MB/s e cada "
    "especialista tem 10 MB, quantos milissegundos leva para ler 8 especialistas?",
]


def call(prompt: str) -> dict:
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "options": OPTS}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def toks_per_s(count, dur_ns):
    return count / (dur_ns / 1e9) if count and dur_ns else 0.0


results = []
print("=== warmup ===")
warm = call("Responda apenas: ok")
results.append({"kind": "warmup", "full": warm})
print(f"warmup: eval_count={warm.get('eval_count')} load_duration={warm.get('load_duration')}")

for i, prompt in enumerate(PROMPTS, 1):
    r = call(prompt)
    entry = {
        "kind": "prompt",
        "n": i,
        "prompt": prompt,
        "response": r.get("response"),
        "prompt_eval_count": r.get("prompt_eval_count"),
        "prompt_eval_duration": r.get("prompt_eval_duration"),
        "eval_count": r.get("eval_count"),
        "eval_duration": r.get("eval_duration"),
        "load_duration": r.get("load_duration"),
        "full": r,
    }
    results.append(entry)
    print(f"\n=== prompt {i} ===")
    print("PROMPT:", prompt)
    print("RESPOSTA:", (r.get("response") or "").strip())
    print(f"prompt tok/s = {toks_per_s(r.get('prompt_eval_count'), r.get('prompt_eval_duration')):.2f}"
          f" | eval tok/s = {toks_per_s(r.get('eval_count'), r.get('eval_duration')):.2f}"
          f" | prompt_eval_count={r.get('prompt_eval_count')}"
          f" | prompt_eval_duration={r.get('prompt_eval_duration')} ns"
          f" | eval_count={r.get('eval_count')}"
          f" | eval_duration={r.get('eval_duration')} ns"
          f" | load_duration={r.get('load_duration')} ns")

print("\n=== ollama ps ===")
ps_out = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
print(ps_out)

print("=== memoria dos processos ollama (pid rss KB vsz KB) ===")
allps = subprocess.run(["ps", "-eo", "pid,rss,vsz,comm"], capture_output=True, text=True).stdout
mem = [l for l in allps.splitlines() if "ollama" in l]
print("\n".join(mem))

with open("baseline_results.json", "w", encoding="utf-8") as f:
    json.dump({"model": MODEL, "options": OPTS, "ollama_ps": ps_out.strip(),
               "processos_ollama": mem, "results": results}, f, ensure_ascii=False, indent=2)
print("\nJSON salvo em baseline_results.json")
