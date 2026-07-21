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
script_dir = os.path.dirname(os.path.abspath(__file__))
h5_path = os.path.join(script_dir, "model.h5")
model = tf.keras.models.load_model(h5_path)

# convertendo o modelo para tensorflow lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# aplicando a técnica de otimização
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# convertendo o modelo 
tflite_model = converter.convert()

# salvando o modelo otimizado na pasta do projeto
os.makedirs(script_dir, exist_ok=True) # garantindo que a pasta vai existir
tflite_path = os.path.join(script_dir, "model.tflite")
open(tflite_path, 'wb').write(tflite_model) # salvando o modelo otimizado
