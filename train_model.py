import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D

keras.utils.set_random_seed(77)

classes = [
    'Футболка / майка', 'Брюки', 'Свитер', 'Платье', 'Пальто',
    'Сандалии', 'Рубашка', 'Кроссовки', 'Сумка', 'Ботильоны'
]

# Загружаем датасет Fashion-MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Нормализуем яркость в [0, 1] и добавляем канал цвета
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# One-hot encoding для 10 классов
y_train_cat = utils.to_categorical(y_train, 10)
y_test_cat = utils.to_categorical(y_test, 10)


def build_cnn():
    return Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2), strides=2),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2), strides=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax'),
    ])


model = build_cnn()
model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

model.fit(
    x_train, y_train_cat,
    batch_size=32,
    epochs=10,
    validation_split=0.2,
    callbacks=[keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2, restore_best_weights=True
    )],
)

scores = model.evaluate(x_test, y_test_cat, verbose=0)
print(f'Тестовая accuracy на Fashion-MNIST: {scores[1]:.4%}')
print(f'Тестовая loss на Fashion-MNIST: {scores[0]:.4f}')

project_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(project_dir, 'artifacts')
os.makedirs(out_dir, exist_ok=True)
model.save(os.path.join(out_dir, 'garment_cnn.keras'))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(os.path.join(out_dir, 'garment_cnn.tflite'), 'wb') as f:
    f.write(tflite_model)
