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

### Protocolo sugerido do baseline

- **Prompts padrão:** fixar 3 prompts curtos (p.ex. "resuma", "raciocínio passo a passo", "código") com temperatura fixa.
- **Medição:** `ollama run <model> --verbose` ou API `/api/generate` com `stream:false` para pegar `eval_count`, `eval_duration`, `prompt_eval_count` (calcular tokens/s = eval_count / eval_duration).
- **RAM pico:** monitorar com `ollama ps` (coluna size) + `/usr/bin/time -v` quando aplicável.
- **Logs:** salvar em `experiments/logs/baseline-ollama/<modelo>.log`.
