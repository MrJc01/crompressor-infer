# experiments/ — Laboratório do Crompressor-Infer

Esta pasta abriga os **scripts isolados** de validação matemática e benchmarks
que rodam **antes** de qualquer integração no motor Go+C do projeto.

## O que entra aqui

- **Benchmarks de I/O** do SSD SATA (leituras aleatórias/sequenciais de 1–8 MiB)
  — pré-requisito do Caminho A.
- **Validação matemática** (NumPy/torch) de codebook → lookup table contra os
  pesos reais de uma camada — pré-requisito do Caminho B.
- **Protótipos de formato** (deltas esparsos de 1–2 bits, cabeçalhos, índices)
  medindo tamanho e custo de decodificação.
- **Logs de baseline** (Ollama e afins) em `logs/`.

## Regras

1. Script = uma pergunta mensurável. O resultado vai para o diário em
   `docs/02-registro-experimentos.md` com números, nunca "pareceu mais rápido".
2. Nada daqui integra o motor (Go+C) sem experimento concluído com comparação
   contra o baseline do Ollama.
3. Nenhum script daqui toca no clone de referência `crompressor/` (somente leitura).
4. Grandes artefatos (modelos, dumps de pesos) **não** são versionados aqui.

## Estrutura sugerida

```
experiments/
├── README.md            (este arquivo)
├── io-ssd/              (benchmarks de disco — Caminho A)
├── codebook-lut/        (validação NumPy de lookup tables — Caminho B)
├── delta-sparse/        (protótipos de deltas 1–2 bits — Caminhos A/C)
└── logs/
    └── baseline-ollama/
```
