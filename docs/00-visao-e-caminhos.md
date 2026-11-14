# Crompressor-Infer — Visão e Caminhos (Fase 0)

> **Nota de contexto:** o código de referência do `crompressor` original vive em
> `crompressor/` (subpasta, clone intacto — usar apenas como leitura). Tudo que
> for desenvolvido neste fork será criado na raiz (`docs/`, `experiments/`,
> `cmd/`, `pkg/`, etc.), sem modificar o clone de referência.

## Objetivo do fork

Rodar LLMs sem GPU dedicada, usando CPU (com SIMD), RAM limitada (14 GiB) e
armazenamento local (SSD SATA 512 GB), reduzindo ao máximo o tráfego de disco e
de memória no caminho crítico da inferência.

O `crompressor` original prova que compressão agressiva permite armazenar pesos
muito menores em disco. Este fork pergunta: **dá para inferir diretamente dessa
representação comprimida, sem descomprimir tudo para RAM a cada passo?**

Baseline de comparação: **Ollama** (llama.cpp), no mesmo hardware, com o mesmo
modelo/quantização quando possível. Nenhum experimento vale algo sem número
comparado ao baseline — registrado em `docs/01-ambiente-e-baseline.md`.

## Os 3 caminhos a testar

### Caminho A — MoE Base + Delta Esparso

**Ideia central:** para modelos MoE, a maioria dos pesos nunca é usada em um
token dado. Guardar a base comprimida no SSD e aplicar apenas o delta esparso
(índices + valores) do conjunto de experts realmente roteado para o token.

- Lê do SSD só os blocos necessários (streaming direto com `pread`, sem FUSE).
- Sem XOR/Zstd no hot path: o delta esparso já é a compressão (poucos
  elementos, índices + fp16/int8).
- Meta: reduzir leitura de SSD por token em 10x+ vs. carregar pesos densos,
  mantendo a RAM baixa.
- Risco: overhead de acesso aleatório a blocos pequenos em **SSD SATA**
  (banda limitada ~500 MB/s); medir IOPS/throughput de leituras de 1–8 MiB
  primeiro (`experiments/`).

### Caminho B — Codebook + Lookup Table

**Ideia central:** evoluir o CodebookLinear do crompressor. Em vez de
descomprimir codebook → matrizes → multiplicar, materializar uma **lookup table
por camada/bloco** que permite computar o produto direto da tabela (residente
na cache L1/L2 da CPU), eliminando a maioria das multiplicações no hot path.

- A tabela é construída uma vez (offline) por grupo de pesos; a inferência
  vira gather + soma, amigável a SIMD (**AVX2 + FMA** presentes neste
  hardware — sem AVX-512).
- Tráfego de RAM quente reduzido: a tabela fica residente em L1/L2 entre
  chamadas repetidas de blocos.
- Meta: tokens/s competitivos com Ollama Q4/Q8 em CPU, com disco ocupando
  2–4x menos.
- Risco: tamanho da tabela × precisão (perplexity pós-tabela precisa ser
  medida contra o modelo fp16/gguf original).

### Caminho C — Híbrido (Codebook em RAM + Deltas esparsos de 1–2 bits do SSD)

**Ideia central:** fundir B + A. Os pesos "quentes" (bases + codebook) ficam
residindo na RAM em formato lookup table; os **deltas esparsos de 1–2 bits**
(rotas MoE, quantização residual) são puxados sob demanda do SSD só para os
experts/blocos ativos do token.

- RAM alvo: base pequena residente (alguns GiB) + streaming sob demanda.
- SSD só entrega deltas comprimidos de baixa entropia (1–2 bits/elemento
  antes de cabeçalhos), baratos de decodificar.
- Meta: melhor trade-off tokens/s × RAM × disco dos três caminhos.
- Risco: complexidade de duas representações coerentes (consistência de
  codebook entre base e delta).

## Restrições aprendidas do crompressor original

Por que **não** usar estas técnicas no caminho crítico da inferência:

1. **XOR (contra arquivos originais/deltas):** exigiria que os dados
   "originais" estivessem presentes ou passíveis de reconstrução no hot path —
   significa ler/decompressar mais dados do que se leria só a representação
   final. XOR é ótimo para armazenar (dedupe/diff), ruim para inferir (a
   reconstrução acontece em cima da hora).
2. **Zstd/fluxos genéricos de compressão:** decodificação por bloco custa
   ciclos de CPU e cache misses no momento exato em que o CPU deveria estar
   fazendo multiplicação/gather. Compressão genérica não "sabe" a estrutura do
   peso (não explora esparsidade de MoE nem codebook); no hot path o custo de
   CPU da descompressão compete diretamente com o custo da própria matemática
   da rede.
3. **FUSE (bazil.org/fuse / go-fuse):** cada leitura de página passa por um
   round-trip userspace↔kernel com transformação na hora; a latência por
   operação explode e o cache de página esconde o custo real até estourar
   RAM. Para streaming de pesos, abrir o arquivo (`pread`) direto e mapear
   blocos manualmente dá controle total de latência e cache.

**Conclusão de Fase 0:** compressão forte fica na **camada de armazenamento**
(formato em disco) e a matemática da inferência deve operar em representações
de baixo custo direto (lookup tables, deltas esparsos, quantizações de 1–2
bits) — nunca em descomprimir genericamente antes de calcular.

## Próximos passos (Fase 1)

1. Baseline Ollama com modelo pequeno (ex.: um MoE 1–3B em Q4) — tabela em
   `docs/01-ambiente-e-baseline.md`.
2. Benchmark de I/O do SSD SATA (leituras aleatórias 1–8 MiB) — Caminho A.
3. Validação matemática do codebook→lookup table em NumPy com uma camada
   real — Caminho B.
4. Só depois disso: criar o motor Go+C **na raiz deste repositório**, usando
   o clone `crompressor/` apenas como referência de leitura.
