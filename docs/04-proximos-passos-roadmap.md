# Crompressor-Infer — Próximos Passos e Roadmap de Engenharia

> Documento oficial de consolidação e direcionamento do projeto. Sintetiza o estado validado na prática entre os experimentos **EXP-001 e EXP-015** e detalha com rigor técnico os **5 caminhos futuros** para a continuidade da pesquisa e do desenvolvimento.

---

## 1. Estado Atual Consolidado (O que já está validado e funcional)

O projeto provou em hardware de consumo real (AMD Ryzen 5 5600GT 6C/12T, 14 GiB RAM, SSD SATA) a viabilidade de inferir modelos MoE sem GPU, com baixa pegada de memória e desacoplamento do carregamento monolítico tradicional:

1. **Motor Dinâmico C puro + Go (CGO, AVX2/FMA/F16C, OpenMP):**
   - Execução em 6 threads físicos sem dependência de PyTorch, CUDA, ROCm ou bibliotecas externas em runtime.
   - Headers dinâmicos lidos em tempo de execução via `.cromllm` (suporte transparente a variações de `hidden_size`, `intermediate_size`, contagem de camadas, contagem de experts e cabeças de atenção/KV).
   - Estabilidade CGO garantida por `runtime.LockOSThread()` e `debug.SetGCPercent(-1)`, eliminando contenção do time OpenMP e pausas do garbage collector durante a geração de tokens.

2. **Formato `.cromllm` e Subsistema de I/O em Disco:**
   - Alinhamento uniforme a 4 KiB (páginas de memória física e blocos de disco).
   - Leitura de pesos via **`O_DIRECT`** contornando a page cache do Linux (`pread()` assíncrono alinhado) para previsibilidade determinística de memória.
   - Cache LRU de experts em RAM com dimensionamento configurável via flag `--cache-experts`.
   - **Prefetch Assíncrono (`pthread`, EXP-012):** sobreposição de I/O de disco com computação de camadas MoE, com bloqueio/pinning no eviction do LRU para evitar race conditions.
   - **Batch Prefill GEMM (`q4_matmul_batch`, EXP-012):** aceleração do processamento inicial de prompt em lote contíguo.

3. **Formato Híbrido `C1_TOPK_K8` (`format_id = 2`, EXP-013A, EXP-013B e EXP-015):**
   - Quantização compacta combinando **Codebook K-Means em cache L1** ($d=4, K=256$, exatamente 4.096 bytes) + **Top-8 Deltas esparsos por grupo de 64** com escala por grupo em FP16 ($[s_{\text{base}}, \alpha]$).
   - Taxa de compressão de **3,50 bits/peso** nos experts internos e Q4_G64 nas camadas extremas.
   - Kernels de cálculo direto em C com desempacotamento vetorial AVX2, sem necessidade de desquantizar matrizes completas na memória.
   - **Resultados empíricos comprovados em 3.3B MoE (EXP-015):** redução de **8,34 GB de tráfego de leitura do SSD SATA (-22,1%)** nos modos sob restrição de cache.
   - **Resiliência a OOM comprovada sob cgroups (`MemoryMax=900M`):** o `crom-infer` opera continuamente com 640 experts em streaming, enquanto o `llama-server` do Ollama exige 2,29 GB fixos e é sumariamente eliminado pelo kernel (`SIGKILL` / OOM killer).

4. **Multi-Fine-Tune em CPU com `.cromdelta` 1-bit (EXP-014):**
   - Conceito exclusivo de servir variantes finetunadas aplicando deltas esparsos de 1-bit diretamente no acumulador FP32: $y = W_{\text{base}} \cdot x + \Delta_{\text{1-bit}} \cdot x$.
   - Redução do delta Instruct para **204,0 MB** (vs 829,5 MB do modelo completo, economia de 75,4%) com perplexidade preservada (PPL 61.20 vs 63.14 do Q4 puro).
   - Formato binário `.cromdelta` (293,8 MB com tokenizador e embeddings) transmitido e sincronizado via rede local por subcomandos Go nativos (`serve-delta` e `pull-delta`) com streaming em blocos de 1 MiB e validação ponta a ponta por hash SHA-256.

---

## 2. Caminhos Futuros de Engenharia

Com os fundamentos matemáticos e a infraestrutura de I/O validados, os próximos passos do projeto estruturam-se em 5 avenidas complementares de pesquisa e implementação:

```
                               ┌────────────────────────────────────────────────────────┐
                               │            Crompressor-Infer (Motor Base)             │
                               │        Go 1.22 + C / AVX2 / O_DIRECT / LRU 4K          │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
          ┌───────────────────────┬───────────────────────┼───────────────────────┬───────────────────────┐
          │                       │                       │                       │                       │
          ▼                       ▼                       ▼                       ▼                       ▼
    [CAMINHO 1]             [CAMINHO 2]             [CAMINHO 3]             [CAMINHO 4]             [CAMINHO 5]
  Modelos > RAM           Calibração AWQ/         Especulação SSD         GEMM Prefill            Ponte Ecossistema
  (Colibrì: > 14 GiB)     imatrix (< 3 bits)      (Draft RAM + Verify)    (BLAS Otimizado)        (Plugin CROM P2P)
  Qwen-30B / OLMoE        Proteger Salientes      1B draft -> 30B SSD     Fechar gap com          Injeção em llama.cpp
  Teto cabo SATA          Meta: 2,0–2,5 bpw       2x a 3x tok/s SSD       Ollama (250+ t/s)       ou Colibrì C
```

---

### Caminho 1: O Desafio dos Modelos Maiores que a RAM Física (> 14 GiB — Território Colibrì)

**Objetivo Central:** Demonstrar o valor do projeto no cenário onde o modelo **não pode ser mantido na memória física**, forçando soluções convencionais de `mmap` a sofrer *page thrashing* catastrófico.

- **Alvos Arquiteturais:**
  - `Qwen/Qwen3-30B-A3B` (30,5 bilhões de parâmetros totais, 128 experts, ~3,3B ativos por token, ~17 GB em Q4).
  - `allenai/OLMoE-1B-7B-0924` (7 bilhões de parâmetros totais, 64 experts, ~1,3B ativos, ~4,5 GB em Q4).
- **Desafios e Requisitos de Engenharia:**
  - **Suporte a Novas Primitivas:** adaptar `crom_infer.c` para suportar arquiteturas MoE com projeções separadas `gate_proj`, `up_proj` e `down_proj` (SwiGLU padrão), além de normalização de chaves e consultas (`qk_norm`).
  - **Vocabulários Extensos:** gerenciar matrizes de embedding e LM-head de 152k tokens (`Qwen`) com paginação sob demanda ou decomposição linear em disco para não inflar a porção Hot residente.
  - **Caracterização do Teto de Hardware:** medir a velocidade limite imposta pela largura de banda contínua do SSD SATA (~450–500 MB/s práticos) em face do número de experts roteados por token ($k \times L$).
  - **Métrica de Sucesso:** alcançar geração estável e sem vazamentos de memória em modelos de 30B rodando com apenas 4 a 6 GiB de RAM alocados ao cache de experts.

---

### Caminho 2: Calibração de Ativações e Hessiana (AWQ / imatrix) para Compressão < 3 bits

**Diagnóstico Atual:** No EXP-006 e no EXP-013A, tentar reduzir a quantização direta para `C1_TOPK_K4` (~3,0 bits/peso) sem calibração prévia provocou degradação sensível da perplexidade (PPL subiu para ~85 no perfil híbrido). A seleção ingênua de centróides por K-Means trata todos os parâmetros com o mesmo peso estatístico, penalizando pesos salientes (*outliers*) críticos para a estabilidade dos estados ocultos.

- **Solução Planejada:**
  - **Coleta de Estatísticas de Ativação:** executar um conjunto padrão de calibração (ex: 512 sequências da Wikitext-2 / C4) em FP16/BF16 para acumular a matriz de covariância das ativações ou o módulo médio das entradas $s_x = \mathbb{E}[|x|]$.
  - **Ponderação por Importância (Salience Weighting):** minimizar a função de perda ponderada durante o treino do Codebook:
    $$\mathcal{L} = \| (W - \hat{W}) \cdot \text{diag}(s_x) \|_F^2$$
  - **Isolamento de Outliers Críticos:** canalizar os $K$ pesos com maior erro quadrático ponderado para a lista de deltas esparsos de alta precisão (FP16 ou 2-bit residual), mantendo os centróides do codebook restritos à distribuição densa de menor variância.
- **Métrica de Sucesso:** atingir **2,0 a 2,5 bits por peso** nos experts intermediários com degradação de PPL inferior a 5% em relação ao BF16, cortando o volume de leitura do disco por token em **40% a 50%**.

---

### Caminho 3: Decodificação Especulativa no SSD (Draft Model em RAM + Verificação no Disco)

**Fundamento Físico:** Como medido empiricamente no EXP-012, ler um bloco de expert de 884 KiB do SSD SATA consome praticamente o mesmo tempo (~2,2 a 2,6 ms de latência de leitura e busca no barramento) quer ele processe 1 token individual ou um lote contíguo de 4 a 6 tokens no GEMM. A decodificação sequencial autoregressiva é severamente limitada pela latência de I/O por passo.

- **Arquitetura Proposta:**
  - **Draft Model Residente (em RAM):** o modelo leve `granite-3.1-1b-a400m` reside integralmente em memória (871 MB RSS), decodificando de forma autônoma a **~50 tok/s** para gerar uma sequência proposta de $K$ tokens especulativos: $(\hat{t}_1, \hat{t}_2, \dots, \hat{t}_K)$.
  - **Target Model em Streaming (no SSD):** o modelo principal de 3.3B (ou 30B) recebe o lote de $K$ tokens em um único passo de prefill/verificação em lote (`crom_forward_batch`).
  - **Amortização de I/O:** os experts necessários para os $K$ tokens são lidos uma única vez do disco e computados conjuntamente via GEMM vetorial.
- **Métrica de Sucesso:** multiplicar o throughput do modo streaming de SSD em **2x a 3x** (elevando de ~3,2 tok/s para **7,0–10,0+ tok/s** em 50% de RAM).

---

### Caminho 4: Aceleração do Prefill com Kernels GEMM Otimizados (Fechar o Gap com o Ollama)

**Diagnóstico Atual:** O prefill do `crom-infer` evoluiu expressivamente no EXP-012 (saltando de 52 para ~90 tok/s no 1B e atingindo ~41 tok/s no 3B). No entanto, o Ollama/llama.cpp atinge de 250 a 580 tok/s porque utiliza rotinas BLAS altamente vetorizadas em blocos de registradores que maximizam a taxa de ocupação dos pipelines FMA da CPU.

- **Soluções em Engenharia de Baixo Nível:**
  - **Micro-kernel GEMM com Register Blocking $6 \times 16$:** reescrever o laço interno de `q4_matmul_batch` utilizando um esquema de empacotamento em ladrilhos (*tiling*) que mantenha 12 a 14 acumuladores YMM ativos simultaneamente, reduzindo o tráfego com o cache L1 de dados.
  - **Integração de Backend BLAS Leve (Header-only):** avaliar a inclusão de bibliotecas de álgebra linear sem dependências runtime pesadas (ex: kernels INT8/FP16 inspirados em `gemmlowp` ou `tinyBLAS`) dedicadas à projeção de atenção e aos roteadores MoE durante o prompt eval.
- **Métrica de Sucesso:** atingir **150+ tok/s de prefill no 3B** e **300+ tok/s no 1B** em CPU com 6 threads, eliminando a discrepância no tempo de ingestão do prompt.

---

### Caminho 5: Ponte de Ecossistema e P2P CROM

**Fundamento Estratégico:** A maturidade do projeto permitiu validar componentes pioneiros — em particular o formato `.cromdelta` de 1-bit e a sincronização distribuída por blocos criptográficos com hash SHA-256. Em vez de reescrever dezenas de variantes de transformadores e kernels manuais de CPU do zero, podemos desacoplar essas inovações e oferecê-las ao ecossistema existente.

- **Iniciativas Planejadas:**
  - **Biblioteca CROM-Delta Integrável:** exportar a lógica de `delta_matvec_add` e do cabeçalho `.cromdelta` como uma biblioteca C pura e autocontida (`libcromdelta.a` / `libcromdelta.h`), capaz de ser embutida diretamente como backend de pesos adicionais no `llama.cpp` ou no `Colibrì`.
  - **Daemon P2P em Go:** evoluir os comandos `serve-delta` e `pull-delta` para um protocolo de fofoca (*gossip protocol*) ou distribuição descentralizada baseada em DHT (Distributed Hash Table), permitindo que nós locais compartilhem deltas de especialistas instantaneamente conforme novas tarefas de fine-tuning sejam disponibilizadas.
- **Métrica de Sucesso:** carregar e rodar um delta `.cromdelta` gerado pelo nosso pipeline dentro de um runtime externo como o `llama.cpp` sem conversão offline de pesos.

---

## 3. Matriz Comparativa e Ordem de Execução

| Caminho | Complexidade | Retorno Esperado | Pré-requisitos | Prioridade |
|---|---|---|---|---|
| **Caminho 1 (Modelos > RAM)** | Alta | Validação definitiva contra mmap em hardware extremo | Parser para SwiGLU / gate-up separados | **1** |
| **Caminho 3 (Especulação SSD)** | Média-Alta | Multiplicação de 2x a 3x no decode de SSD | Batch Prefill funcional (EXP-012) | **2** |
| **Caminho 2 (Calibração < 3 bits)** | Alta | Redução de 40-50% de I/O em disco com PPL alta | Pipeline Python com PyTorch/imatrix | **3** |
| **Caminho 4 (Aceleração Prefill)** | Média | Fechar gap de ingestão vs Ollama | Micro-kernels AVX2 register blocking | **4** |
| **Caminho 5 (Ponte Ecossistema)** | Baixa-Média | Adoção e interoperabilidade externa | Estabilização final do formato .cromdelta | **5** |
