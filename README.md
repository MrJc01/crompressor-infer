# crompressor-infer

Motor experimental de inferência de LLM **sem GPU**, escrito em **Go 1.22 + C
(CGO/AVX2/FMA/F16C)**, focado em rodar modelos **MoE** com RAM limitada e SSD
SATA. Os pesos vivem num formato próprio **`.cromllm`** alinhado a 4 KiB,
pensado para leitura direta com **`O_DIRECT`** + **Cache LRU de experts** — ou
seja, inferir lendo do disco apenas os experts que o roteador escolhe.

> Projeto de pesquisa/engenharia. O código de referência do `crompressor`
> original fica em `crompressor/`, uma subpasta **externa e ignorada** pelo
> versionamento (apenas leitura). Todo o desenvolvimento deste fork fica na
> raiz.

## Modelo e hardware de referência

- Modelo: `granite-3.1-1b-a400m-instruct`, empacotado em
  `artifacts/granite-1b.cromllm` (Q4_g64, ~791 MiB em disco).
- Hardware: AMD Ryzen 5 5600GT (6 núcleos / 12 threads, AVX2+FMA+F16C, **sem
  GPU**), 14 GiB RAM, SSD SATA. **6 threads físicos** é o ponto ótimo.

## Duelo oficial EXP-010

4 prompts padrão, chat template do Granite, greedy (`temperature=0`),
`--max-tokens 128 --threads 6`. Médias dos 4 prompts:

| Engine | decode tok/s | prefill tok/s | RAM | Observações |
|---|---|---|---|---|
| PyTorch BF16 | 3,81 | — | 3.482 MB (pico) | referência numérica |
| Ollama Q8_0 | 41,32 | 587,8 | ~1.740 MB | baseline llama.cpp |
| **crom-infer RAM (768)** | **47,78** | 52,9 | **871 MB** | **100 %** cache hit |
| **crom-infer SSD (576)** | **15,06** | 8,9 | **709 MB** | **90,9 %** hit (`O_DIRECT`) |

Os tokens do modo SSD (576) são **idênticos** aos do modo RAM (768) nos 4
prompts. Detalhes: `experiments/logs/crom-infer/duelo_ollama.json`.

## Estrutura de pastas

```
cmd/crom-infer/        CLI Go (flags, prefill/decode, --json, --tokenize-only)
internal/engine/       wrapper CGO do motor C + leitura do tokenizer do header
internal/tokenizer/    tokenizador BPE/ByteLevel nativo (validado vs HuggingFace)
engine/                motor C (crom_infer.c/.h) e CLI C de referência
tools/                 check_tokenizer.py, duelo_ollama.py, empacotador .cromllm
docs/                  visão, ambiente/baseline, registro de experimentos, formato, roadmap
experiments/           benchmarks, logs e baselines (Ollama / PyTorch)
artifacts/             modelo empacotado .cromllm (não versionado)
crompressor/           clone de referência EXTERNO, ignorado pelo git (somente leitura)
```

## Build, validação e uso

Requisitos: Go 1.22+, GCC com OpenMP, e o modelo em
`artifacts/granite-1b.cromllm`.

```bash
# build (biblioteca C + binário Go)
make all

# validação (tokenizador Go vs HuggingFace + testes Go)
make test

# geração com chat template (RAM Full)
./bin/crom-infer --prompt "Explain a hash table in one sentence:" \
  --chat --max-tokens 128 --threads 6 --cache-experts 768

# modo streaming (75% RAM, O_DIRECT)
./bin/crom-infer --prompt "Explique tabela hash." \
  --chat --max-tokens 128 --threads 6 --cache-experts 576

# saída JSON com métricas (decode/prefill tok/s, RSS, cache hit, ssd_stream)
./bin/crom-infer --prompt "Hi" --chat --max-tokens 32 --threads 6 \
  --cache-experts 768 --json

# apenas tokenizar (sem carregar pesos)
./bin/crom-infer --tokenize-only --prompt "Explain a hash table in one sentence:"

# duelo oficial
python3 tools/duelo_ollama.py
```

## Documentação

- [docs/00 — Visão e caminhos](docs/00-visao-e-caminhos.md)
- [docs/01 — Ambiente e baseline](docs/01-ambiente-e-baseline.md)
- [docs/02 — Registro de experimentos](docs/02-registro-experimentos.md)
- [docs/03 — Formato `.cromllm`](docs/03-formato-cromllm.md)
- [docs/04 — Próximos passos / roadmap (EXP-011+)](docs/04-proximos-passos-roadmap.md)

## Estado atual

Validado até o **EXP-010**: motor C+AVX2/FMA com `O_DIRECT` e LRU, CLI Go+CGO
estável (~59 tok/s no modo RAM), tokenizador BPE nativo 100 % fiel ao
HuggingFace e duelo oficial contra Ollama/PyTorch. Próximos marcos (modelo maior
que a RAM, prefetch assíncrono, `C1_v2` a 3,10 bits, prefill em lote, deltas
P2P) estão descritos em `docs/04`.
