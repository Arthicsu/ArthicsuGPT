from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import os


def train_and_save_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2), strides=2),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2), strides=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Обучение модели...")
    history = model.fit(x_train, y_train_cat,
                        batch_size=32,
                        epochs=5,
                        validation_split=0.2,
                        verbose=1)

    scores = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"Точность на тестовых данных: {scores[1] * 100:.2f}%")

    model.save('/content/mnist_model.h5')
    print("Модель сохранена как 'mnist_model.h5'")

    return model

if __name__ == "__main__":
    train_and_save_model()