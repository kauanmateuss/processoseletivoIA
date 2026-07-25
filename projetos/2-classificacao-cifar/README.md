# Projeto 2 — Classificação CIFAR-10

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar imagens coloridas** em 10 categorias de objetos e animais (avião, automóvel, pássaro, gato, cervo, cachorro, sapo, cavalo, navio, caminhão), e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

Este projeto tem uma diferença importante em relação a uma classificação de dígitos: as imagens são **coloridas (RGB)** e visualmente mais complexas, o que torna a tarefa de classificação genuinamente mais difícil — por isso **data augmentation** é um requisito obrigatório aqui, não opcional.


## 📝 Relatório do Candidato

👤 **Nome Completo:** Kauan Mateus Vidal de Brito

### 1️⃣ Resumo da Arquitetura do Modelo

A construção deste modelo foi um processo iterativo. Inicialmente, realizei testes de treinamento com uma rede mais simples(como uma rede com apenas uma camada Conv2D) e diferentes hiperparâmetros(como batchsize de 10 e 16; learnig rate de 0,003 e 0,001 além de menos Data Augmentation). Após os treinamentos e analise de métricas importantes como perda e acurácia, pude notar que o modelo ficava apenas na casa dos 60% de acurácia, dando a entender que o modelo tivesse com baixa capacidade de extração de características nas imagens.

Com o intuito de atingir pelo menos 80% de acurácia, optei construir uma rede não muito complexa que conta com 3-4 blocos(como foi sugerido nos requisitos do projeto), onde cada bloco possui duas camadas Conv2D e duas camadas de BatchNormalization. A configuração final consiste em:

  - **Data Augmentation:** Integrada diretamente no modelo sequencial, aplicando transformações como: Inversão horizontal (RandomFlip), Rotação de 10% (RandomRotation), Zoom de 10% (RandomZoom), Translação (RandomTranslation) e Contraste (RandomContrast).

  - **Extração de Características (CNN):** 3 blocos profundos. Cada bloco possui 2 camadas Conv2D (com padding="same" para preservar as bordas), seguidas por camadas de BatchNormalization (para acelerar a convergência e estabilizar o treino) e finalizadas com MaxPooling2D para redução de dimensionalidade. Os filtros aumentam progressivamente (32 -> 64 -> 128).

  - **Classificador Final:** Uma camada Flatten seguida por uma camada oculta Dense interpretadora de 256 neurônios. Em seguida, um Dropout agressivo de 50% garante a generalização, e a saída ocorre por uma camada Dense de 10 neurônios com ativação Softmax.

Os treinamentos foram realizados utilizando **early stopping** com patience variáveis(3, 5, 10) baseado na perda de validação(como foi pedido nos requisitos).


### 2️⃣ Bibliotecas Utilizadas

Foram utilizadas as seguintes bibliotecas para realização do projeto:
 
  - **TensorFlow / Keras (2.15.0):** Framework principal utilizado para construção e treinamento do modelo. A versão 2.15.0 foi fixada propositalmente para garantir a compatibilidade.

  - **NumPy:** Para manipulações numéricas e arrays padrão do Python.

  - **Scikit-Learn:** Utilizada especificamente pela função train_test_split, garantindo a separação estratificada (stratify) dos dados de validação.

Essas especificações também está disponível no nosso arquivo requirements.txt do projeto

### 3️⃣ Técnica de Otimização do Modelo

Para realizar a otmização do modelo treinado utilizei a técnica de **Quantização** que foi ensinada durante a capacitação no modulo de Otimização de Modelos em Sistemas Embarcados.
A técnica utilizada foi a **Dynamic Range Quantization**(Quantização de Faixa Dinâmica), na qual é acionada através da flag tf.lite.Optimize.DEFAULT. Essa abordagem reduz os pesos do modelo para inteiros de 8 bits (int8), diminuindo o tamanho do arquivo e acelerando a execução, mas mantendo entradas e saídas em float32. Essa técnica costuma trazer ganhos de eficiência com perda mínima de acurácia.

Mesmo optando por utilizar a técnica que foi vista na capacitação, realizei pesquisas sobre outras técnicas de Quantização como a:
 
  - **Full Integer Quantization:** É uma evolução da técnica que utilizamos no projeto, ela converte tudo(pesos e ativações) para inteiro de 8 bits

### 4️⃣ Resultados Obtidos

Após realizar vários treinamentos refinando os valores dos hiperparâmetros, o modelo alcançou métricas consistentes e provou não sofrer de overfitting severo:
 
  - **Acurácia de validação Final:** Conseguimos 81%
 
  - **Tamanho do modelo original:** O modelo model.h5 tem um tamanho de 9,9MB
 
  - **Tamanho do modelo otimizado:** O modelo model.tflite tem um tamanho de 836,9kB

A aplicação da quantização reduziu o modelo de 9,9MB para 836,9kB, resultando em uma compressão de aproximadamente 11,8 vezes, o que é ideal para o uso em Edge AI.

### 5️⃣ Comentários Adicionais (Opcional)

Bom, acredito que a maior dificuldade encontrada no projeto foi em relação ao conflito das dependências entre o meu ambiente local e o ambiente de testes do GitHub Actions. Por esse motivo, precisei fixar a versão do TensorFlow para 2.15.0 no requirements.txt para garantir a correta desserialização dos pesos do modelo.

Toda a base teórica adquirida na capacitação do Pnaat, somada às minhas outras capacitações em Visão computacional, me deu a visão sistêmica necessária para iterar sobre a rede. Pude realizar vários treinamentos com redes mais simples e testar hiperparâmetros distintos, chegando na seguinte configuração ideal para bater a meta de 80% de acurácia:
  
  - **Tamanho do Batch:** 64

  - **Número de Épocas:** 40 (controladas pelo Early Stopping)

  - **Taxa de Aprendizado (Learning Rate):** 0.001. Testei com taxas um pouco mais altas, porem o modelo não conseguia chegar a 80% de acurácia com apenas 40 epocas.

  - **Paciência do Early Stopping:** Foi utilizado 5. Esse valor foi escolhido tecnicamente para evitar que o treino fosse interrompido precocemente por flutuações normais na perda de validação, mas curto o suficiente para impedir que o modelo começasse a decorar os dados

  - **Arquitetura:** 2 camadas Conv2D por bloco convolucional.

Por ser uma CNN sequencial clássica, o modelo carece de conexões residuais. Tentar torná-lo significativamente mais profundo para buscar acurácias acima de 90% esbarraria no problema do Vanishing Gradient (onde os gradientes ficam muito próximos de zero, fazendo com que as camadas iniciais parem de aprender).

Com pesquisas e revendo partes da capacitação para sanar algumas dúvidas específicas durante o desenvolvimento do projeto consegui relembrar muita coisa e até aprender coisas novas e algumas curiosidades. 

### 6️⃣ Exemplo de Inferência

Abaixo está a saida gerada ao executar o script run_inference.py no terminal utilizando o modelo otimizado model.tflite

![Imagem do resultado da inferência](https://private-user-images.githubusercontent.com/124316642/626665522-238e2ba9-f7c1-484e-9766-1976827d843b.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODQ5ODQ0MDcsIm5iZiI6MTc4NDk4NDEwNywicGF0aCI6Ii8xMjQzMTY2NDIvNjI2NjY1NTIyLTIzOGUyYmE5LWY3YzEtNDg0ZS05NzY2LTE5NzY4MjdkODQzYi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNzI1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDcyNVQxMjU1MDdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1jOTI1N2I2ZTRjNzQ4MGI3YzMxNDUxZDI5MTk5OTdjNzkwMTc3MDIwZGQ4NjVlMjQ4OTQ2ODdlM2U5NThiY2ZmJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcifQ.QQp-bXwCBT5USrFZiYft0zOO4OPeAsWv-EDXKz53n9o)

No geral, o modelo apresenta uma alta confiança nas predições. Pude notar, no entanto, que às vezes ele confunde a classe bird com airplane. E de acordo com pesquisas que realizei, isso é um falso positivo comum no dataset CIFAR-10, já que ambas as classes frequentemente compartilham o mesmo tipo de background, o que pode enganar os filtros de extração se as bordas do objeto central não forem tão nítidas.