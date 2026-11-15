# Crompressor-Infer — Ambiente e Baseline

Coletado em **13/11/2026** (Fase 0). Máquina: `jcrom` — **Linux nativo, sem WSL2**.

## Hardware

| Item | Valor | Observação |
|---|---|---|
| SO | Linux Mint 22.3 (Zena), kernel 7.0.0-42-generic, x86_64 | Linux nativo (não WSL2) |
| CPU | AMD Ryzen 5 5600GT with Radeon Graphics | 6 núcleos / 12 threads, até ~4,67 GHz |
| Instruções vetoriais | **AVX2 + FMA** (também SSE4.2, AES, SHA-NI, VAES) | **Sem AVX-512** |
| RAM | 14 GiB total (~6,3 GiB disponível na coleta) + 2 GiB swap | Restrição dura para modelos residentes |
| Disco do projeto | `/dev/sda2` ext4 — **SSD SATA "KA-SSD-512"** (476,9 GB, não rotativo) | **~49 GB livres (90% usado)** — restrição de espaço; banda SATA (~500 MB/s máx teórico), não NVMe |
| GPU | Apenas integrada: AMD Radeon Vega (Cezanne), compartilhada com RAM | **Sem GPU dedicada, sem CUDA, sem ROCm** |

## Ferramentas instaladas

| Ferramenta | Versão | Observação |
|---|---|---|
| Go | **1.22.2** | ⚠️ `crompressor/go.mod` exige **1.25.7** — o clone de referência não compila com este Go ainda |
| GCC | 13.3.0 (Ubuntu) | Para o futuro motor C |
| Clang | 18.1.3 | Alternativa de compilador C |
| Make | 4.3 | OK |
| CMake | **não instalado** | Pendência |
| Python | 3.12.3 | |
| Libs Python | torch, numpy, transformers, safetensors, gguf — todas OK | Já instaladas no ambiente |
| uv | 0.12.15 | OK |
| Ollama | **0.30.7** — daemon ativo (`ollama serve` PID 1385, serviço `active`) | **Nenhum modelo baixado** (`ollama list` vazio) |

## Pendências de ambiente (para Fase 1)

- Atualizar/instalar Go 1.25.x OU ajustar planos para não compilar o clone.
- `cmake` (futuras builds C).
- Espaço em disco: só **~49 GB livres** — cuidado ao baixar modelos (um MoE 8B Q4 ≈ 5 GB; manter margem).
- Baixar o primeiro modelo de baseline no Ollama.

## Baseline do Ollama (a preencher)

Registrado aqui com **o mesmo hardware acima**. Um experimento só é aceito se
comparar contra uma linha desta tabela.

| # | Modelo | Quantização | Tokens/s (geração) | Tokens/s (prompt eval) | RAM pico | Disco ocupado | Prompts padrão | Log de saída |
|---|---|---|---|---|---|---|---|---|
| # | Modelo | Quantização | Tokens/s (geração) | Tokens/s (prompt eval) | RAM pico | Disco ocupado | Prompts padrão | Log de saída |
|---|---|---|---|---|---|---|---|---|
| 1 | granite3.1-moe:1b | (interna Ollama, ~1,4 GB em disco) | 39,5–45,0 (média ~41,3) | 501–695 | ~1,6 GiB RSS (`llama-server`) + 34 MB (`ollama serve`) | 1,4 GB | 4 prompts fixos (EXP-002) | `experiments/logs/baseline-ollama/baseline_results.json` |
| 2 | **crom-infer Q4_g64 (RAM 768)** | Q4_g64 (idx u8 + scales f32, grupo 64), experts residentes | **47,8** (média 4 prompts) | **52,9** | **871 MB RSS** | **791 MiB** (`artifacts/granite-1b.cromllm`) | 4 prompts fixos (EXP-010) | `experiments/logs/crom-infer/duelo_ollama.json` |
| 3 | **crom-infer Q4_g64 (SSD 576)** | Q4_g64 + streaming `O_DIRECT` (75% RAM, LRU) | **15,1** (média 4 prompts) | **8,9** | **709 MB RSS** (cache_hit **90,9%**, `ssd_stream` **2971,9 MB**) | **791 MiB** (`artifacts/granite-1b.cromllm`) | 4 prompts fixos (EXP-010) | `experiments/logs/crom-infer/duelo_ollama.json` |
| 4 | **granite3.1-moe:3b** (Ollama) | Q4_K_M (interna Ollama, ~2,0 GB em disco) | **33,33** (média 4 prompts) | **254,87** | **2.272,8 MB RSS** (`llama-server`) | **2,0 GB** | 4 prompts fixos (EXP-015) | `experiments/logs/crom-infer/exp015_granite3b_duel.json` |
| 5 | **crom-infer 3B Q4 (RAM 1280)** | Q4_G64 (3.3B MoE, 1280 experts residentes) | **22,78** (média 4 prompts) | **41,52** | **2.188,0 MB RSS** | **1.883,2 MiB** (`artifacts/granite-3b-q4.cromllm`) | 4 prompts fixos (EXP-015) | `experiments/logs/crom-infer/exp015_granite3b_duel.json` |
| 6 | **crom-infer 3B Híbrido (RAM 1280)** | Híbrido Q4 + C1_TOPK_K8 (3.50 bpw) | **13,14** (média 4 prompts) | **17,30** | **1.925,5 MB RSS** | **1.620,8 MiB** (`artifacts/granite-3b-hybrid.cromllm`) | 4 prompts fixos (EXP-015) | `experiments/logs/crom-infer/exp015_granite3b_duel.json` |
| 7 | **crom-infer 3B Híbrido (SSD 640)** | Híbrido C1_K8 + Streaming SSD 50% RAM | **3,18** (média 4 prompts) | **8,44** | **1.369,5 MB RSS** (economia I/O **-22,1%** vs Q4) | **1.620,8 MiB** (`artifacts/granite-3b-hybrid.cromllm`) | 4 prompts fixos (EXP-015) | `experiments/logs/crom-infer/exp015_granite3b_duel.json` |

### Protocolo sugerido do baseline

- **Prompts padrão:** fixar 3 prompts curtos (p.ex. "resuma", "raciocínio passo a passo", "código") com temperatura fixa.
- **Medição:** `ollama run <model> --verbose` ou API `/api/generate` com `stream:false` para pegar `eval_count`, `eval_duration`, `prompt_eval_count` (calcular tokens/s = eval_count / eval_duration).
- **RAM pico:** monitorar com `ollama ps` (coluna size) + `/usr/bin/time -v` quando aplicável.
- **Logs:** salvar em `experiments/logs/baseline-ollama/<modelo>.log`.
