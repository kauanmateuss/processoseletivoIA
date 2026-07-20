import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 2 — Otimização do Modelo (CIFAR-10)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

# insira seu código aqui

# carregando o modelo que treinamos
model = tf.keras.models.load_model("model.h5")

# convertendo o modelo para tensorflow lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# aplicando a técnica de otimização
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# convertendo o modelo 
tflite_model = converter.convert()

# salvando o modelo otimizado
open('model.tflite', 'wb').write(tflite_model)
