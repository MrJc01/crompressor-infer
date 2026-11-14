#!/usr/bin/env python3
"""Valida o tokenizador do ./bin/crom-infer Go contra HuggingFace.

Compara `ids_go == ids_hf` para o prompt cru e para os 4 prompts de chat
padrao (chat template Granite com add_generation_prompt=True).

Uso:
    python3 tools/check_tokenizer.py [--bin ./bin/crom-infer]
"""
import argparse
import json
import subprocess
import sys

MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"

RAW_PROMPT = "Explain a hash table in one sentence:"

CHAT_PROMPTS = [
    "Explique em 3 frases curtas como funciona uma tabela hash.",
    "What is a Mixture-of-Experts neural network? Answer concisely in 3 sentences.",
    "Escreva apenas uma função em C que soma dois vetores float de tamanho n. Sem texto extra.",
    "Resolva passo a passo e dê o resultado final: Se um disco lê a 250 MB/s e cada especialista tem 10 MB, quantos milissegundos leva para ler 8 especialistas?",
]


def go_tokenize(binary: str, prompt: str, chat: bool) -> list:
    cmd = [binary, "--tokenize-only", "--prompt", prompt]
    if chat:
        cmd.append("--chat")
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)["ids"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="./bin/crom-infer")
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover
        print(f"SKIP: transformers indisponivel ({exc})")
        return 2

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

    cases = [("raw", RAW_PROMPT, False, tok.encode(RAW_PROMPT, add_special_tokens=False))]
    for i, prompt in enumerate(CHAT_PROMPTS, 1):
        hf_ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        cases.append((f"chat{i}", prompt, True, hf_ids))

    all_match = True
    for name, prompt, chat, hf_ids in cases:
        go_ids = go_tokenize(args.bin, prompt, chat)
        match = go_ids == hf_ids
        all_match = all_match and match
        print(f"[{name}] match={match} len_go={len(go_ids)} len_hf={len(hf_ids)}")
        if not match:
            print(f"  ids_go={go_ids}")
            print(f"  ids_hf={hf_ids}")
        else:
            print(f"  ids_go==ids_hf={go_ids}")

    print("RESULT:", "ALL MATCH" if all_match else "MISMATCH")
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
