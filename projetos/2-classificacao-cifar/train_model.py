import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Projeto 2 — Classificação CIFAR-10
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset CIFAR-10 via tf.keras.datasets.cifar10
#   2. Normalizar as imagens para [0, 1] (shape (32, 32, 3))
#   3. Separar um conjunto de validação
#   4. Incluir data augmentation (ex: layers.RandomFlip, RandomRotation, RandomZoom)
#      aplicada ao conjunto de treino
#   5. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   6. Treinar com EarlyStopping monitorando a perda de validação
#   7. Exibir a acurácia de validação final no terminal
#   8. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

# insira seu código aqui

# Carregando o dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


# Normalizando as imagens para o intervalo [0,1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# Fazendo a separacao do conjunto de validacao
# validação é 20% do conjunto de treino
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.2)


# data augmentation para conjunto de treino
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),   # vira as imagens horizontalmente
    layers.RandomRotation(0.1),     # rotaciona as imagens em até 10%
    layers.RandomZoom(0.1),     # aplica zoom aleatorio nas imagens em até 10%
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),  # aplicando translação aleatoria 
    layers.RandomContrast(0.1)  # aplicando contraste aleatorio
], name="data_augmentation")


# Construindo a CNN
model = keras.Sequential()

model.add(layers.Input(shape=(32, 32, 3)))  # camada de entrada com shape (32, 32, 3)

# aplicando data augmentation
model.add(data_augmentation)

# bloco 1
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Bloco 2
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# bloco 3
model.add(layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))


model.add(layers.Flatten()) # Transforma em vetor unidimensional
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.5))  # Regularização com Dropout de 50%
model.add(layers.Dense(10, activation="softmax")) # camada de saida

# Fazendo a compilação do modelo
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Mostrando o resumo do modelo que definimos
model.summary()

# Definindo o callback de EarlyStopping para monitorar a perda
# Vai parar se a perda não melhorar por 10 epocas e vai restaurar os pesos do melhor modelo
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)

# Fazendo o treinamento do modelo
history = model.fit(
    x=x_train, 
    y=y_train,
    validation_data=(x_val, y_val),
    epochs=30,
    batch_size=32,
    shuffle=True,
    callbacks=[early_stopping]
)

# Exibindo a acurácia de validacao
val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
print(f"Acurácia de validação final: {val_accuracy:.4f}")

# caminho para salvar o modelo treinado
script_dir = os.path.dirname(os.path.abspath(__file__))

os.makedirs(script_dir, exist_ok=True)  # cria a pasta se não existir

h5_path = os.path.join(script_dir, "model.h5")

# salvando o modelo treinado como "model.h5" na pasta atual
model.save(h5_path)
