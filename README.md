# Crompressor-Infer — Motor Experimental de Inferência MoE sem GPU em Go + C/AVX2 com Streaming Direto de SSD e Deltas de 1-bit

Motor de alta performance desenvolvido para executar modelos de linguagem baseados em **Mistura de Especialistas (MoE)** em hardware de consumo comum, **sem GPU**, com restrição severa de memória RAM e utilizando **SSD SATA** de baixo custo.

O projeto combina uma CLI e subsistema de rede em **Go 1.22+** com um runtime numérico de baixo nível em **C puro (AVX2/FMA/F16C/OpenMP)**, operando sobre formatos binários próprios (`.cromllm` e `.cromdelta`) alinhados em páginas de 4 KiB para acesso direto via **`O_DIRECT`**, contornando a page cache do sistema operacional.

---

## 1. Hardware e Ambiente de Validação

Todos os benchmarks foram executados em máquina física nativa (sem virtualização ou WSL2):
- **Processador:** AMD Ryzen 5 5600GT (6 núcleos físicos / 12 threads, AVX2 + FMA + F16C, até 4,67 GHz)
- **Memória RAM:** 14 GiB físicos DDR4
- **Armazenamento:** SSD SATA comum 512 GB (`/dev/sda2`, barramento SATA ~500 MB/s teóricos)
- **Aceleração Gráfica:** **Nenhuma GPU dedicada** (sem CUDA, sem ROCm, sem Metal)
- **Configuração de Threads:** 6 threads físicos (ponto de saturação ótima dos barramentos)

---

## 2. Resumo Executivo dos Resultados Reais

### 2.1 Comparativo Oficial: Granite-1B (MoE 1.3B) e Granite-3.3B (MoE 3.3B)

Resultados médios obtidos sobre a suíte de 4 prompts padrão (programação, raciocínio passo a passo, explicação didática e concisão conceitual) com temperatura zero (`greedy`):

| Modelo / Arquitetura | Motor / Modo de Execução | Cache de Experts | RAM Utilizada (RSS) | Prefill (tok/s) | Decode (tok/s) | Volume I/O Streaming SSD | Economia I/O vs Q4 |
|---|---|---|---|---|---|---|---|
| **Granite-1B** (24L, 32 exp) | PyTorch BF16 (Referência) | N/A (RAM) | 3.482 MB | — | 3,81 | 0 MB | — |
| **Granite-1B** (24L, 32 exp) | Ollama Q8_0 (`llama.cpp`) | N/A (RAM) | 1.740 MB | **587,82** | 41,32 | 0 MB | — |
| **Granite-1B** (24L, 32 exp) | **crom-infer Q4 (RAM)** | 768 (100%) | **871 MB** | 52,92 | **47,78** | 0 MB | — |
| **Granite-1B** (24L, 32 exp) | **crom-infer Q4 (SSD Streaming)**| 576 (75%) | **709 MB** | 8,93 | **15,06** | 2.971,9 MB | — |
| **Granite-1B** (24L, 32 exp) | **crom-infer Híbrido C1_K8 (SSD)**| 576 (75%) | **699 MB** | 10,75 | **18,00** | 2.450,2 MB | **-17,6%** |
| **Granite-3.3B** (32L, 40 exp) | Ollama Q4_K_M (`llama.cpp`) | N/A (RAM) | 2.272 MB | **254,87** | **33,33** | 0 MB | — |
| **Granite-3.3B** (32L, 40 exp) | **crom-infer Q4 (RAM)** | 1280 (100%) | **2.188 MB** | 41,52 | 22,78 | 0 MB | Baseline |
| **Granite-3.3B** (32L, 40 exp) | **crom-infer Híbrido C1_K8 (RAM)**| 1280 (100%) | **1.925 MB** | 17,30 | 13,14 | 0 MB | — |
| **Granite-3.3B** (32L, 40 exp) | **crom-infer Q4 (SSD Streaming)**| 640 (50%) | **1.378 MB** | 10,72 | 3,19 | 37.762,4 MB | Baseline 50% |
| **Granite-3.3B** (32L, 40 exp) | **crom-infer Híbrido C1_K8 (SSD)**| 640 (50%) | **1.369 MB** | 8,44 | **3,18** | **29.423,5 MB** | **-8.339,0 MB (-22,1%)** |

---

### 2.2 Prova de Fogo sob Restrição de RAM: `MemoryMax=900M`

Em teste cego de contenção de recursos gerenciado pelo kernel Linux via `systemd-run --user --scope -p MemoryMax=900M`:
- **Ollama (`llama-server`):** aloca estaticamente **2.291 MB de RSS**. Quando submetido à cota máxima de 900 MB, é **sumariamente encerrado pelo kernel via `SIGKILL` (código 137, OOM killer)** antes de emitir um único token.
- **`crom-infer`:** com `--cache-experts 640`, aloca inicialmente apenas **833 MB** e gerencia o pool de memória dinamicamente com substituição LRU e prefetch em background. O processo completou todas as requisições com geração contínua e sem estouro de pilha.
- **Ganho do Formato Híbrido C1_TOPK_K8:** sob 50% de RAM, manteve taxa idêntica de decodificação (~3,18 tok/s) transferindo **8,34 GB a menos** do SSD SATA (-22,1% de desgaste e contenção de barramento).

---

### 2.3 Multi-Fine-Tune e Sincronização de Deltas por Rede (`.cromdelta` 1-bit)

O projeto demonstrou um paradigma inédito de adaptação de domínio para modelos MoE:
- **Cálculo Direto sem Materialização:** $y = W_{\text{base}} \cdot x + \Delta_{\text{1-bit}} \cdot x$, executado em CPU via instruções `_mm256_fmadd_ps` com desempacotamento vetorial de sinais.
- **Economia Extrema de Armazenamento:** o arquivo de delta Instruct para o Granite-1B ocupa **204,0 MB** de pesos (vs 829,5 MB do modelo Instruct completo), uma economia de **75,4%**. O pacote `.cromdelta` consolidado (incluindo embeddings e tokenizador) totaliza **293,8 MB** (63% menor que o modelo base).
- **Qualidade Superior a Q4:** a perplexidade (PPL) do modelo Base + $\Delta_{\text{1-bit}}$ atingiu **61,20** na Wikitext-2 (melhor do que os **63,14** do modelo quantizado em Q4 tradicional).
- **Distribuição P2P por Chunks:** subcomandos nativos em Go (`serve-delta` e `pull-delta`) que transferem o delta por rede local em chunks de 1 MiB com validação criptográfica automática via **SHA-256**.

---

## 3. Arquitetura e Formatos Binários

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Formato .cromllm                                │
├───────────────┬──────────────────────────────────┬──────────────────────────┤
│ Header (4K)   │ Magic, Hiperparâmetros, Offsets  │ Dinâmico (1B, 3B, etc.)  │
├───────────────┼──────────────────────────────────┼──────────────────────────┤
│ Tokenizer     │ tokenizer.json embutido          │ Parser Go nativo BPE     │
├───────────────┼──────────────────────────────────┼──────────────────────────┤
│ Hot Block     │ Embeddings (FP16), Normas (FP32),│ Residente contínuo       │
│               │ Roteadores (FP32), Atenção (Q4)  │ em RAM                   │
├───────────────┼──────────────────────────────────┼──────────────────────────┤
│ Expert Dir    │ Tabela de Offsets, Sizes e Formatos (24B por entrada)       │
├───────────────┼──────────────────────────────────┼──────────────────────────┤
│ Expert Pool   │ Experts gravados em blocos contíguos de 4 KiB               │
│               │ - format_id = 1: Q4_G64 (grupos de 64, scales FP32)         │
│               │ - format_id = 2: C1_TOPK_K8 (Codebook L1 4K + Top-8 deltas) │
└───────────────┴─────────────────────────────────────────────────────────────┘
```

- **Motor C (`engine/crom_infer.c`):** micro-kernels vetoriais desenvolvidos à mão em AVX2/FMA/F16C com OpenMP, suporte a prefetch assíncrono com threads POSIX (`pthread`) e I/O não bloqueante com `O_DIRECT`.
- **Tokenizador Go (`internal/tokenizer/`):** reimplementação fiel do `GPT2TokenizerFast` (ByteLevel, split de regex, merges por rank) com 100% de paridade token a token contra a biblioteca `transformers` da HuggingFace.

---

## 4. Compilação e Uso

### 4.1 Requisitos
- Linux x86_64 com suporte a AVX2, FMA e F16C
- Compilador GCC 13+ com suporte a OpenMP (`-fopenmp`)
- Go 1.22 ou superior

### 4.2 Build e Testes
```bash
# Compilar biblioteca C e gerar binário Go unificado (./bin/crom-infer)
make all

# Executar testes unitários do tokenizador e engine Go
go test ./...
```

### 4.3 Exemplos de Inferência na CLI

```bash
# Inferência padrão com chat template em RAM total (100% dos experts em cache)
./bin/crom-infer --model artifacts/granite-3b-q4.cromllm \
  --prompt "Explain what a hash table is in simple terms:" \
  --chat --max-tokens 128 --threads 6 --cache-experts 1280

# Inferência com restrição de memória (50% RAM + Streaming O_DIRECT de SSD)
./bin/crom-infer --model artifacts/granite-3b-hybrid.cromllm \
  --prompt "Explain what a hash table is in simple terms:" \
  --chat --max-tokens 128 --threads 6 --cache-experts 640 \
  --repeat-penalty 1.15 --json

# Inferência Multi-Fine-Tune (Modelo Base + Delta 1-bit de Instruct)
./bin/crom-infer --model artifacts/granite-1b-base.cromllm \
  --delta artifacts/granite-1b-instruct.cromdelta \
  --prompt "Write a Python function to check if a number is prime:" \
  --chat --max-tokens 64 --threads 6

# Servir delta via rede local HTTP (porta 8080)
./bin/crom-infer serve-delta --delta artifacts/granite-1b-instruct.cromdelta --port 8080

# Baixar e validar delta por rede com verificação SHA-256
./bin/crom-infer pull-delta --url http://127.0.0.1:8080 --out artifacts/recebido.cromdelta
```

---

## 5. Estrutura do Repositório

```
cmd/crom-infer/         Ponto de entrada da CLI em Go (flags, subcomandos e timers)
engine/                 Motor numérico em C puro (crom_infer.c, crom_infer.h, kernels AVX2)
internal/engine/        Camada de integração CGO e gerenciamento de contexto
internal/tokenizer/     Tokenizador BPE nativo com chat template
tools/                  Scripts auxiliares (empacotamento .cromllm, check de tokenizador, duelos)
experiments/            Registros brutos de benchmarks e saídas JSON de referência
docs/                   Documentação arquitetural e diário de experimentos
artifacts/              Modelos compilados e deltas (armazenamento local, fora do git)
```

---

## 6. Documentação Detalhada

- **[docs/00 — Visão e Caminhos](docs/00-visao-e-caminhos.md):** Fundamentos teóricos e hipóteses do projeto.
- **[docs/01 — Ambiente e Baselines](docs/01-ambiente-e-baseline.md):** Especificação do hardware e tabela de baselines oficiais.
- **[docs/02 — Registro de Experimentos](docs/02-registro-experimentos.md):** Diário técnico e relatórios detalhados do EXP-001 ao EXP-015.
- **[docs/03 — Formato `.cromllm`](docs/03-formato-cromllm.md):** Especificação binária byte a byte dos arquivos de modelo e diretórios.
- **[docs/04 — Próximos Passos e Roadmap](docs/04-proximos-passos-roadmap.md):** Guia detalhado dos 5 caminhos futuros (Modelos > RAM / Colibrì, AWQ/imatrix < 3 bits, Especulação no SSD, GEMM Prefill e Ponte P2P).
