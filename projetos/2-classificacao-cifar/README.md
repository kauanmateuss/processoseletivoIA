# Projeto 2 — Classificação CIFAR-10

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar imagens coloridas** em 10 categorias de objetos e animais (avião, automóvel, pássaro, gato, cervo, cachorro, sapo, cavalo, navio, caminhão), e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

Este projeto tem uma diferença importante em relação a uma classificação de dígitos: as imagens são **coloridas (RGB)** e visualmente mais complexas, o que torna a tarefa de classificação genuinamente mais difícil — por isso **data augmentation** é um requisito obrigatório aqui, não opcional.

## 🎯 Conjunto de Dados

Dataset **CIFAR-10**, disponível diretamente via `tf.keras.datasets.cifar10` (não é necessário download manual). 60.000 imagens 32x32 coloridas, 10 classes.

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset CIFAR-10 via TensorFlow
- Split explícito treino/validação
- **Data augmentation** aplicada ao conjunto de treino, usando camadas do Keras
  (ex: `RandomFlip("horizontal")`, `RandomRotation`, `RandomZoom`) incorporadas ao
  modelo ou ao pipeline de treino
- Construção de uma CNN com 3-4 blocos convolucionais (`Conv2D` + `BatchNormalization`
  + `MaxPooling2D`) seguida de `Dropout`
- Treinamento com **early stopping** baseado na perda de validação
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

> 💡 Se você aplicar a augmentation de outra forma (ex: pré-processamento manual em
> `tf.data`), tudo bem — apenas descreva isso claramente no relatório, já que a
> correção automática busca primeiro por camadas de augmentation no próprio modelo.

> 💡 CIFAR-10 é mais difícil que MNIST/Fashion-MNIST para uma CNN simples treinada
> rapidamente em CPU — não se preocupe se a acurácia ficar bem abaixo de 90%. O
> importante é o pipeline completo funcionar corretamente.

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/2-classificacao-cifar/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 32x32, 3 canais (RGB), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 25-30, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Generalização** — uso adequado de data augmentation
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Kauan Mateus Vidal de Brito

### 1️⃣ Resumo da Arquitetura do Modelo

Descreva a arquitetura da CNN implementada em `train_model.py` e a estratégia de data augmentation utilizada.

A construção deste modelo foi um processo iterativo. Inicialmente, realizei testes/treinamento com uma rede
simples(como utilizando apenas uma camada Conv2D) e diferente hiperparâmetros(como batchsize = 10, 16; learnig rate = 0,003, 0,001 e menos aumentação de dados). Após os treinamentos e analise de métricas importantes como perda e acurácia, pude notar que o modelo ficava apenas na casa dos 60% de acurácia, onde podia ser que o modelo tivesse com baixa capacidade de extração de características.

Com o intuito de atingir pelo menos 80% de acurácia, optei construir uma rede não muito complexa que conta com 3-4 blocos(como foi sugerido nos requisitos do projeto), onde cada bloco possui duas camadas Conv2D e duas camadas de BatchNormalization. A configuração final consistem em:

  - **Data Augmentation:** Integrada diretamente no modelo sequencial, aplicando transformações como: Inversão horizontal (RandomFlip), Rotação de 10% (RandomRotation), Zoom de 10% (RandomZoom), Translação (RandomTranslation) e Contraste (RandomContrast).

  - **Extração de Características (CNN):** 3 blocos profundos. Cada bloco possui 2 camadas Conv2D (com padding="same" para preservar as bordas), seguidas por camadas de BatchNormalization (para acelerar a convergência e estabilizar o treino) e finalizadas com MaxPooling2D para redução de dimensionalidade. Os filtros aumentam progressivamente (32 -> 64 -> 128).

  - **Classificador Final:** Uma camada Flatten seguida por uma camada oculta Dense interpretadora de 256 neurônios. Em seguida, um Dropout agressivo de 50% garante a generalização, e a saída ocorre por uma camada Dense de 10 neurônios com ativação Softmax.

Foi utilizado 

### 2️⃣ Bibliotecas Utilizadas

Liste as principais bibliotecas utilizadas, preferencialmente com suas versões.

### 3️⃣ Técnica de Otimização do Modelo

Explique qual técnica foi utilizada para otimizar o modelo em `optimize_model.py`.

### 4️⃣ Resultados Obtidos

Informe a acurácia de validação obtida e o tamanho dos arquivos `model.h5` e `model.tflite`.

### 5️⃣ Comentários Adicionais (Opcional)

Dificuldades encontradas, decisões técnicas importantes, limitações do modelo, aprendizados durante o desafio.

### 6️⃣ Exemplo de Inferência

Cole a saída do terminal ao rodar `run_inference.py` (predito vs. real para as 5+ amostras), e comente brevemente se houve algum caso interessante (acerto ou erro) entre as amostras testadas.
