# Crompressor-Infer — Registro de Experimentos

Diário padronizado de experimentos. **Regra:** nenhum experimento entra aqui
sem métricas numéricas e comparação contra o baseline do Ollama
(`docs/01-ambiente-e-baseline.md`). Scripts ficam em `experiments/`.

---

## [EXP-###] — <título curto>

**Data:** AAAA-MM-DD
**Fase:** 0/1/2/…
**Caminho:** A (MoE Base + Delta Esparso) / B (Codebook + Lookup Table) / C (Híbrido) / infra (ambientes, I/O, sem caminho)
**Status:** planejado / em execução / concluído / abandonado

### Objetivo

<o que este experimento tenta responder, em 1–3 frases>

### Hipótese

<métrica esperada vs. métrica atual, quantificada — ex.: "leitura de SSD por token cai 10x vs. baseline">

### Código/Script

- Script: `experiments/<subpasta>/<script>` (ou caminho do código)
- Como rodar: <comando exato>
- Requisitos: <libs/versões/hardware envolvido>

### Comparação contra Ollama

| Métrica | Baseline Ollama | Este experimento | Δ |
|---|---|---|---|
| <ex.: tokens/s> | | | |
| <ex.: SSD lido por token (MB)> | | | |
| <ex.: RAM pico (GiB)> | | | |

Baseline usado: linha #_ da tabela em `docs/01-ambiente-e-baseline.md`.

### Métricas numéricas

<valores brutos: latências, throughput, tamanhos, precisão/perplexidade — sem arredondar demais>

### Conclusão

<cita os números acima. Aceito/rejeitado? Próximo passo consequente (novo experimento, mudança de plano ou integração no motor)>

---

## [EXP-001] — Benchmark físico do SSD SATA (leituras 4 KiB–4 MiB)

**Data:** 13/11/2026
**Fase:** 1
**Caminho:** infra (I/O — pré-requisito do Caminho A)
**Status:** concluído

### Objetivo

Medir a banda e a latência reais do SSD SATA (KA-SSD-512) para leituras
sequenciais e aleatórias, em blocos de 4 KiB a 4 MiB, com `O_DIRECT`,
para saber quanto tráfego o Caminho A consegue sustentar por token.

### Hipótese

Banda de ~250–450 MB/s (limite SATA) para blocos grandes; leituras de 4 KiB
aleatórias limitadas por IOPS (~8 mil), o que penaliza deltas esparsos muito
fragmentados — logo, ler blocos ≥ 1 MiB é quase "grátis" (sequencial ≈ aleatório).

### Código/Script

- Script: `experiments/io-ssd/bench_ssd.c` (C, `gcc -O2 -Wall`)
- Como rodar: `gcc -O2 -Wall -o bench_ssd bench_ssd.c && ./bench_ssd` (dentro de `experiments/io-ssd/`)
- Requisitos: arquivo de teste de 256 MiB criado e removido pelo próprio programa; `O_DIRECT` suportado pelo ext4

### Comparação contra Ollama

Não aplicável (experimento de I/O). Referência indireta: Ollama carrega modelos
por mmap/leitura sequencial — ver comparativo em Conclusão.

### Métricas numéricas (O_DIRECT, CLOCK_MONOTONIC, pread alinhado a 4096)

| Bloco | Modo | MB/s | Latência média (ms) | IOPS |
|---|---|---|---|---|
| 4 KiB | sequencial | 41,5 | 0,094 | 10 614,7 |
| 4 KiB | aleatório | 31,8 | 0,123 | 8 151,0 |
| 64 KiB | sequencial | 200,0 | 0,312 | 3 200,2 |
| 64 KiB | aleatório | 140,7 | 0,444 | 2 251,0 |
| 256 KiB | sequencial | 255,1 | 0,980 | 1 020,5 |
| 256 KiB | aleatório | 207,1 | 1,207 | 828,3 |
| 1 MiB | sequencial | 349,3 | 2,863 | 349,3 |
| 1 MiB | aleatório | 344,5 | 2,903 | 344,5 |
| 4 MiB | sequencial | 428,1 | 9,343 | 107,0 |
| 4 MiB | aleatório | 449,0 | 8,909 | 112,2 |

### Conclusão

A partir de 1 MiB, leitura aleatória custa **o mesmo** que sequencial
(344,5 vs 349,3 MB/s; 449,0 vs 428,1 MB/s) — o SSD SATA satura a banda e o
padrão de acesso deixa de importar. Em 4 KiB, porém, só há 8–10 mil IOPS
(31–41 MB/s): ler deltas esparsos em blocos minúsculos despenca a banda para
~8% do teto. **Implicação para o Caminho A:** agrupar deltas em blocos de
≥ 1 MiB (granularidade de expert/bloco de camadas) para ficar no teto de
~350–450 MB/s; leituras de 4 KiB só valem para metadados/índices.
Teto físico confirmado: ~450 MB/s (coerente com SATA III).
Aceito como premissa de projeto do Caminho A.

---

## [EXP-002] — Baseline oficial do MoE `granite3.1-moe:1b` no Ollama

**Data:** 13/11/2026
**Fase:** 1
**Caminho:** infra (baseline de referência para todos os caminhos)
**Status:** concluído

### Objetivo

Estabelecer o baseline numérico de tokens/s, RAM e latência do Ollama
(llama.cpp, 100% CPU) rodando o modelo MoE `granite3.1-moe:1b`, com
temperature 0, seed 42 e num_predict 128, para comparar contra os
Caminhos A/B/C.

### Hipótese

Em CPU puro (num_gpu=0), o Ollama entrega na casa de 30–50 tok/s de geração
e centenas de tok/s de prompt eval — números que os caminhos A/B/C devem
superar (ou igualar com muito menos disco/RAM) para valerem a pena.

### Código/Script

- Script: `experiments/logs/baseline-ollama/run_baseline.py` (Python 3.12, stdlib)
- Como rodar: `python3 experiments/logs/baseline-ollama/run_baseline.py`
- Requisitos: Ollama 0.30.7 com daemon ativo; modelo `granite3.1-moe:1b` puxado
  (`ollama pull granite3.1-moe:1b`); API `http://localhost:11434/api/generate`

### Comparação contra Ollama

Este experimento **é** o baseline. Configuração: `{"temperature": 0.0,
"seed": 42, "num_predict": 128, "num_gpu": 0}`; 1 warmup + 4 prompts fixos.

Baseline usado: linha #1 da tabela em `docs/01-ambiente-e-baseline.md`.

### Métricas numéricas (por prompt: prompt tok/s | eval tok/s | counts/durações)

| # | Prompt (resumo) | Prompt tok/s | Eval tok/s | prompt_eval (tok / ns) | eval (tok / ns) | load_duration (ns) |
|---|---|---|---|---|---|---|
| 1 | tabela hash | 643,96 | 40,82 | 60 / 93 174 000 | 128 / 3 135 341 000 | 60 122 912 |
| 2 | MoE (EN) | 695,09 | 45,03 | 65 / 93 513 000 | 87 / 1 932 239 000 | 96 221 810 |
| 3 | função C | 510,54 | 39,90 | 68 / 133 192 000 | 128 / 3 208 120 000 | 63 529 530 |
| 4 | 8 experts × 10 MB a 250 MB/s | 501,68 | 39,53 | 103 / 205 311 000 | 128 / 3 238 122 000 | 77 280 492 |

Estado do runtime: `ollama ps` → 1,7 GB, **100% CPU**, contexto 4096;
`llama-server` RSS **1 646 216 KB (~1,6 GiB)**; `ollama serve` RSS 34 756 KB.
Modelo em disco: 1,4 GB.

### Respostas literais (para comparação futura)

**Prompt 1** — "Explique em 3 frases curtas como funciona uma tabela hash."
> Uma tabela hash é um algoritmo de criptografia que armazena dados num formato compacto e facilmente manipulável, geralmente usando bits ou bytes. Funciona assim:
>
> 1. O primeiro passo é convertir o dado em uma sequência de bits ou bytes. Isso pode ser feito usando um algoritmo como SHA-256 ou MD5.
>
> 2. A sequência resultante é então dividida em bits ou bytes, geralmente 128 bits (16 bytes) por dado.
>
> 3
_(cortado em 128 tokens; contém alucinação factual: tabela hash ≠ criptografia)_

**Prompt 2** — "What is a Mixture-of-Experts neural network? Answer concisely in 3 sentences."
> A Mixture-of-Experts (MoE) neural network is an extension of the Expert system concept, where multiple expert models are combined to make predictions or decisions. Each expert model specializes in a specific task and contributes to the overall prediction by combining their outputs through a weighted sum, hence the term "Mixture." This approach enhances model robustness and interpretability compared to traditional neural networks.

**Prompt 3** — "Escreva apenas uma função em C que soma dois vetores float de tamanho n. Sem texto extra."
> Aqui está um exemplo de código C que cria duas vectores float e soma-os:
>
> ```c
> #include <stdio.h>
>
> void sumVectors(float *v1, float *v2) {
>     float sum = 0.0;
>
>     // Iterar sobre os elementos do vetor v1
>     for (int i = 0; i < 3; ++i) {
>         sum += v1[i];
>     }
>
>     // Iterar sobre o elemento do vetor v2
>     for (int i = 0; i < 3; ++
_(cortado em 128 tokens; não obedeceu "sem texto extra"; loop fixo em 3 em vez de n)_

**Prompt 4** — "Resolva passo a passo e dê o resultado final: Se um disco lê a 250 MB/s e cada especialista tem 10 MB, quantos milissegundos leva para ler 8 especialistas?"
> Para resolver esse problema, preciso entender como funciona o tempo de leitura de um disco. O tempo de leitura é diretamente proporcional ao número de megas por segundo (MB/s) que a máquina pode ler. Isso significa que se eu multiplicar 250 MB por segundo por 8 especialistas, obtenho o tempo total em milissegundos para leitura de todos os especialistas.
>
> Primeiro, multiplique 250 pelo número de megas por segundo:
> 2
_(cortado em 128 tokens; resposta cortada antes do resultado final — gabarito: 8 × 10 MB = 80 MB → 80/250 s = 320 ms)_

### Conclusão

Baseline estabelecido: **~40–45 tok/s de geração** e **~500–700 tok/s de prompt
eval**, com RAM quente de ~1,6 GiB e o modelo inteiro residente (100% CPU,
1,7 GB pelo `ollama ps`). Respostas têm alucinações e cortes (num_predict 128),
mas servem como referência determinística (seed 42, temperature 0).
**Meta dos caminhos A/B/C:** ≥ 40 tok/s com menos RAM residente e/ou menos
disco por token; caminho A tem teto de ~350–450 MB/s de leitura (EXP-001)
para trazer deltas do SSD. JSON completo em
`experiments/logs/baseline-ollama/baseline_results.json`.

---

## [EXP-003] — Raio-X anatômico do MoE (referência exata BF16) + baseline HF

**Data:** 13/11/2026
**Fase:** 1
**Caminho:** A / B / C (premissas numéricas de todos)
**Status:** concluído

### Objetivo

Carregar a referência exata `ibm-granite/granite-3.1-1b-a400m-instruct`
(bf16, CPU — o mesmo modelo que o Ollama serve quantizado em Q8_0, EXP-002) e
medir: distribuição de parâmetros, similaridade entre os 32 experts por camada,
esparso-idade de `Delta = W − Base`, ganho de codebook K=256 via Delta vs. via
W direto, PPL e tok/s do PyTorch na CPU.

### Hipótese

Se experts fossem próximos uns dos outros (cosseno alto, Delta pequeno),
Base+Delta comprimido barato ia dominar (Caminho A/C). Codebook no Delta
(Caminho B) teria erro menor que no W direto. Nenhuma das duas se confirmou
com força — ver Conclusão.

### Código/Script

- Script: `experiments/logs/baseline-hf/xray_and_baseline.py` (Python 3.12,
  torch 2.12.0+cu130, transformers **4.57.6** — atualizado de 4.45.1 porque
  `granitemoe` exige ≥ 4.47.0; atualização via `pip --user --break-system-packages`,
  conflito registrado apenas com `pyreft` de outro projeto)
- Como rodar: `python3 xray_and_baseline.py` (em `experiments/logs/baseline-hf/`)
- Threads: `torch.set_num_threads(12)` (6c/12t; bf16 gera ~3,8–4,0 tok/s)

### Métricas numéricas

**1) Parâmetros/memória (bf16) — total 1 334 628 352 params / 2 545,6 MB:**

| Componente | Params | MB (bf16) |
|---|---|---|
| Embeddings/LM head (empatados) | 50 334 720 | 96,0 |
| Attention + norms (24 camadas) | 75 547 648 | 144,1 |
| Routers/Gates MoE (24×[32,1024]) | 786 432 | 1,5 |
| **Experts MoE** | **1 207 959 552** | **2 304,0** |

Arquitetura: 24 camadas; vocab 49 155; hidden 1024; inter 512; 32 experts,
**8 ativos por token** (top-8 de 32 = 25% ativo). Por camada:
`input_linear (32,1024,1024)` + `output_linear (32,1024,512)` =
50,3 M params/camada. **Os experts são 90,5% do modelo em disco; a parte
"sempre quente" (embed+attn+norms+router) é só ~237,6 MB.**

**2) Similaridade MoE (Base = média dos 32 experts; média global):**

| Pilha | std(W) | std(Delta) | razão | cos médio expert~Base | esparsidade Δ |
|---|---|---|---|---|---|
| input_linear | 0,01581 | 0,01554 | 1,02 | **0,186** | 21% < 0,25σ; 40% < 0,5σ; 70% < 1σ |
| output_linear | 0,01711 | 0,01684 | 1,02 | **0,177** | 21% / 40% / 70% |

**3) Codebook K=256 (camada 11, input_linear, 100k micro-vetores d=4):**
erro relativo quantizando **W direto = 0,3223**; quantizando **Delta → W_hat =
Base + Δ_hat = 0,3168**; ganho apenas **1,02x**.

**4) Referência exata BF16 (CPU, 12 threads):** PPL = **56,511** (NLL 4,0344,
141 tokens de validação fixa no script); tok/s = **3,6–4,0**; RSS pico do
processo = **3 482 MB**; modelo em disco 2,5 GB (cache HF).

**5) Respostas literais dos 4 prompts (bf16, greedy):**

- P1 (tabela hash): "Uma tabela hash funciona criando um mapa de chaves-valor,
  onde cada chave é um valor único e cada valor é um valor de dados. Ela
  armazena dados em um array, onde cada posição representa um chave. Quando um
  novo valor é inserido, ele é convertido em um chave e acessado na tabela, e
  o valor armazenado é retornado. Se o chave já estiver presente, o valor é
  retornado diretamente, evitando a necessidade de uma busca manual. Isso
  torna a tabela hash muito" (128 tok)
- P2 (MoE EN): "A Mixture-of-Experts neural network is a type of deep learning
  architecture that combines the strengths of multiple expert models to make
  predictions. It consists of multiple expert networks, each specializing in a
  specific task, and a weighted average or voting mechanism to combine their
  outputs, resulting in a more accurate and robust prediction." (68 tok)
- P3 (função C): "Aqui está uma função em C que soma dois vetores float de
  tamanho n: ```c #include <stdio.h> void sumVectors(float *v1, float *v2,
  float *sum) { *sum = 0.0; for (int i = 0; i < 3; i++) { *sum += v1[i] +
  v2[i]; } } int main() { float v1[3] = {1.0, 2.0, 3.0}; float" (128 tok;
  loop fixo em 3 de novo — limitação do modelo, não do runtime)
- P4 (8 experts): "…1. …8 especialistas * 10 MB/especialista = 80 MB. 2. Em
  seguida, convertemos o tempo em segundos, pois a leitura é baseada em MB/s.
  1 MB é igual a 1024 KB," (128 tok, cortado; caminho certo até 80 MB)

JSON completo: `experiments/logs/baseline-hf/hf_baseline.json`;
log integral: `experiments/logs/baseline-hf/xray_run.log`.

### Conclusão

**Premissas dos caminhos A/B/C corrigidas por dado real:**

1. **Experts NÃO são redundantes entre si**: cosseno expert~Base ≈ **0,18** e
   std(Delta)/std(W) ≈ **0,98** (razão 1,02) — os 32 experts são
   estatisticamente quase independentes. Uma "base + delta esparso pequeno"
   (Caminho A/C tal como idealizado) NÃO comprime os experts entre si. O
   espalhamento útil não é entre experts, é **dentro do roteamento**: só 8/32
   experts são usados por token, e os experts somam 90,5% do modelo.
2. **Codebook ganho marginal via Delta** (0,3223 → 0,3168, 1,02x): quase todo
   o erro vem da variância intra-expert, não da média. Codebook de K=256,
   d=4 sozinho não resolve; precisa de blocos maiores (d≥8/16) ou
   quantização residual por camada — seguir para EXP-004 variando d.
3. **Referência exata em PyTorch é ~10x mais lenta que o Ollama** (3,8 vs
   ~40 tok/s) — confirmar a decisão de implementar o motor em Go+C com SIMD,
   e usar a PPL bf16 (**56,511**) e as respostas acima como métrica
   contínua de qualidade para qualquer quantização/codebook futuro.
4. Esparsidade de Delta (70% < 1σ) é modesta — deltas de 1–2 bits não são
   "quase zero", mas um código 2-bit por elemento é plausível se residuais
   forem modelados por camada (Caminho C), não como diferença entre experts.

Próximo passo consequente: EXP-004 — varrer d (4/8/16) e K (256/65536) do
codebook e testar quantização residual (Base global + residuais por expert)
medindo PPL degradada vs. bits/param.

---

## [EXP-004] — Torneio de compressão: 6 variantes (Caminhos A/B/C) em todas as 24 camadas

**Data:** 13/11/2026
**Fase:** 1
**Caminho:** A / B / C (torneio completo sobre os experts MoE)
**Status:** concluído

### Objetivo

Aplicar 6 esquemas de compressão a TODOS os tensores de experts
(input_linear + output_linear das 24 camadas; atenção/embeddings/norms/routers
intactos em bf16), medindo bits/peso, tamanho, erro L2, PPL (mesmo VAL_TEXT do
EXP-003, ref BF16 = 56,511) e gerando os 4 prompts padrão com os 2 vencedores.

### Hipótese

Variantes B (codebook) e C (híbrido codebook+delta esparso) entregariam PPL
razoável com menos bits que o Q4 de referência; A (poda direta de 50% nos
experts) testaria se esparsidade nos pesos preserva a PPL.

### Código/Script

- Script: `experiments/codebook-lut/torneio_compressao.py` (torch 2.12, threads=12)
- Como rodar: `python3 torneio_compressao.py` (backup dos experts em disco +
  `torch.load(mmap=True)` por variante; K-Means com atribuição em chunks de
  163 840 e buffer `out=` reutilizado)
- Notas de execução: 2 correções durante o desenvolvimento — (1) contagem de
  bits do codebook (1 código de 8 bits cobre `d` pesos); (2) `assign` otimizado
  de 56s→21s por tensor grande (buffer pré-alocado evita page-faults).
  RSS pico do processo: **5 638 MB** (acima do alvo de 4,5 GB — pico ocorreu
  na fase de prompts/restauro; 14 GiB no total, sem swap-out detectado).

### Métricas numéricas (experts: 1 207 959 552 params; BF16 = 2 304,0 MB; PPL ref = 56,511)

| Variante | bits/peso | MB experts | Compressão vs BF16 | Erro L2 | PPL | ΔPPL vs BF16 | Tempo |
|---|---|---|---|---|---|---|---|
| BF16 (referência) | 16,0 | 2 304,0 | 1,00x | 0 | 56,511 | — | — |
| Q4_grupo64 | 4,25 | 612,0 | 3,76x | 0,1123 | **63,43** | +6,92 | 25s |
| B1_Codebook_d2_K256 | 4,25 | 612,0 | 3,76x | **0,0917** | 64,82 | +8,31 | 625s |
| B2_Codebook_d4_K256 | 2,25 | 324,1 | 7,11x | 0,3174 | 474,75 | +418,24 | 347s |
| C1_Hibrido_Codebook_DeltaEsparso | 3,55 | 511,3 | 4,51x | 0,1752 | 76,07 | +19,55 | 481s |
| C2_Hibrido_RVQ_d4_K256_K16 | 3,25 | 468,1 | 4,92x | 0,1886 | 89,28 | +32,77 | 384s |
| A_Esparso50_Q4 | 3,25 | 467,3 | 4,93x | 0,2805 | 162,68 | +106,17 | 34s |

Vencedores (menor PPL < 4,5 bits, ≥1 da família B/C): **Q4_grupo64** e
**B1_Codebook_d2_K256**. Velocidade dos prompts quantizados no PyTorch:
**5,9–7,9 tok/s** vs 3,6–4,0 tok/s do BF16 (menos bytes tocados por token).

### Conclusão

1. **Q4 grupo-64 é o campeão de PPL** (63,43, +6,92) — a escala fp16 por grupo
   de 64 preserva outliers de magnitude, e isso importa mais que o erro L2
   global: B1 tem erro menor (0,0917 vs 0,1123) mas PPL PIOR (64,82). Lição
   central para o Caminho B: **codebook "puro" uniformiza o erro e destrói
   outliers** — precisa de escala por grupo misturada com o codebook (ou
   codebook só no residual após remover outlier-scale por linha).
2. **Caminho B extremo (d=4, 2,25 bits) é inviável assim** (PPL 475; err
   0,3174, idêntico ao predito no EXP-003: 0,3168). 2,25 bits só viram
   aceitáveis com esquema RVQ mais profundo ou codebook+outliers.
3. **Caminho C híbrido funciona parcialmente**: C1 (3,55 bits) fica em
   +19,55 de PPL — pior que Q4 4-bit, mas com menos bits; C2 RVQ (3,25) em
   +32,77. Espaço de melhora: escala ternária por grupo e codebook no
   residual com correção de outliers.
4. **Caminho A direto nos pesos é o pior** (poda 50% nos experts → PPL 162,68):
   confirma o EXP-003 — a esparsidade útil dos experts MoE está no
   **roteamento (8/32 ativos por token)**, não nos pesos. A poda de pesos
   não é o caminho; o delta esparso do Caminho A deve vir da diferença
   código↔original em blocos grandes, não de zerar pesos.
5. **Para o motor Go+C:** a meta imediata viável é Q4 grupo-64 (3,76x menos
   disco que BF16, PPL +6,9) e evoluir o codebook com correção de outliers
   para tentar derrubar os 4,25 bits mantendo PPL ≤ Q4.

JSON: `experiments/codebook-lut/torneio_results.json`; log integral:
`experiments/codebook-lut/torneio_run.log`; backup dos experts removido ao
final (disco livre: 45 GB).

---
## [EXP-005] — Trace de localidade do MoE (cache de experts) + duelo de kernels AVX2 (Q4 vs LUT)

**Data:** 2026-11-14 · **Caminho:** A (delta esparso / cache) + B (codebook/LUT) · **Status:** concluído

### Pergunta

1. (T1) Qual a localidade temporal dos experts durante o decode? Quantos
   bytes por token precisam ser lidos do disco se mantivermos um cache
   LRU dos experts em RAM (cap N experts)? Isso valida o Caminho A
   (delta esparso + cache de experts) antes de implementar o motor.
2. (T2) Quanto ganhamos de vazamento em tokens/s trocando o working set
   dos experts de FP32 (4,5 GiB) para formatos comprimidos, com kernels
   AVX2 nativos: Q4 grupo-64 (vencedor do EXP-004), B1 codebook K=256
   com LUT d=2 (vencedor alternativo) e LUT16 (vpxshufb, 4,06 bits)?
   Serve de "speedup máximo teórico" para o motor Go+C antes de SSD.

### Código/Script

- Script T1: `experiments/delta-sparse/trace_moe_cache.py`
  (hook PyTorch em `router.layer` de cada camada; captura os 8 experts
  escolhidos por token durante decode; simulação LRU com caps 4/8/16/24)
- Como rodar: `python3 trace_moe_cache.py` (modelo em cache HF local)
- Script T2: `experiments/codebook-lut/bench_kernels.c` (C99 + AVX2/FMA)
- Como rodar: `gcc -O3 -march=native -fopenmp -Wall -o bench_kernels bench_kernels.c -lm && OMP_NUM_THREADS=12 ./bench_kernels`
- Detalhes: working set sintético com a dimensão exata do modelo
  (24 camadas × 32 experts × (1024×1024 + 512×1024) = 1 207 959 552
  pesos); 10 tokens de decode com 192 matvecs/token (8 experts × 24
  camadas, seleção determinística); validação escalar de cada kernel
  (SIMD vs escalar ~2e-07) e erro de dequantização vs FP32.
- Correções durante o desenvolvimento: (1) pareamento nibble↔peso
  (byte i = peso i low + peso i+32 high, alinhado ao kernel); (2) índice
  do gather da LUT do B1 — cada lane precisa do offset de linha próprio
  `(par)*K + código` (bug silencioso: o micro-teste de validação usava um
  gerador degenerado e mascarava o erro); (3) alocação do `b1_cb` era a
  metade (in+out) — heap overflow em build completo.

### Métricas numéricas

**T1 — trace de localidade (2 prompts, ~450 tokens de decode):**

| Cache (experts) | Hit rate | Bytes lidos do SSD por token (Q4) | tok/s estimados @ 344,5 MB/s |
|---|---|---|---|
| 4 | 4,85% | 182,7 MB | 1,9 |
| 8 | 34,89% | 125,0 MB | 2,8 |
| 16 | 68,69% | 60,1 MB | 5,7 |
| 24 | 89,06% | 21,0 MB | 16,4 |

Conclusão T1: com cap 24/32 (RAM ~7,4 GB no formato Q4 de 1,3B) a
leitura de SSD cai a ~21 MB/token; com cap 8 ainda são 125 MB/token.
O cache de experts só "fecha" perto do working set inteiro — a
recorrência por prompt é alta (89% dos acertos em 24), mas entre
prompts os experts variam pouco (2,7-3,5/32 repetidos por token).
Para o Caminho A, o delta esparso por PROMPT + cache LRU grande é o
que sobrevive; cache pequeno morre no pior caso de conversa nova.

**T2 — duelo de kernels (10 tokens × 192 matvecs; validação SIMD↔escalar ≤ 2e-07):**

| Formato | bits/peso | Bytes/token | 1 thread | 6 threads | 12 threads | Erro dequant |
|---|---|---|---|---|---|---|
| FP32_AVX2 | 32,0 | 1 208,4 MB | 16,1 tok/s | 24,7 tok/s | 18,8 tok/s | — |
| Q4_g64_AVX2 | 4,25 | 170,4 MB | 36,6 tok/s | **171,4 tok/s** | 77,9 tok/s | 0,0697 |
| B1_LUT_d2_K256 | 4,25 | 170,4 MB | 10,4 tok/s | 52,3 tok/s | 22,0 tok/s | 0,0674 |
| LUT16_VPSHUFB | 4,06 | 245,5 MB | 36,0 tok/s | 124,9 tok/s | 61,9 tok/s | 0,0617 |

(números em GB/s efetivo: FP32 29,8 · Q4 29,1 · LUT16 30,6 — todos
no teto de RAM ~30 GB/s; B1 fica em 8,9 GB/s, limitado pelo gather)

### Conclusão

1. **6 threads é o ponto ótimo** no Ryzen 5 5600GT: com 12 threads
   (SMT) TODOS os kernels pioram ~2x — o segundo thread por núcleo só
   disputa banda de RAM. O motor Go+C deve fixar `OMP_NUM_THREADS=6`.
2. **Teto de vazamento confirmado: ~30 GB/s.** FP32, Q4 e LUT16 atingem
   o mesmo GB/s — a diferença de tok/s vem 100% dos bytes tocados por
   token. Com Q4 (3,76x menos bytes), o speedup é 171,4/24,7 = **6,9x**
   sobre o FP32 AVX2, chegando a **171 tok/s sintéticos**.
3. **Q4 grupo-64 é o formato certo para o Caminho B**: melhor tok/s
   absoluto e PPL comprovada no EXP-004 (63,43). LUT16 (vpxshufb) fica
   perto (125 tok/s, erro 0,0617) e é o candidato a "segundo nível"
   se precisarmos de menos bits — o gather do B1 (52 tok/s) fica fora
   da disputa neste hardware sem VP2INTERSECT.
4. **Comparado aos baselines**: Q4@6t = 171 tok/s vs Ollama ~40-45
   tok/s (4x) e vs HF bf16 CPU ~3,8 tok/s (45x) — dentro do estimado,
   porque o Ollama também roda Q8/Q4 mas com overhead de runtime.
5. **Plano do motor**: experts residentes em RAM no formato Q4_g64
   (~612 MB para o modelo 1B inteiro, cabe nos 14 GiB), 6 threads,
   e o disco entra só no fallback de modelos maiores (Caminho A com
   cache LRU cap 24 → ~21 MB/token de SSD, ~16 tok/s adicionais).

JSON/log: `experiments/delta-sparse/moe_cache_trace.json` (T1);
log integral do T2: `experiments/codebook-lut/bench_kernels_run.log`.

---
