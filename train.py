import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import random


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    idx = random.randint(0, len(x_train))
    plt.imshow(x_train[idx], cmap='gray')
    plt.title(f'{class_names[y_train[idx]]} ({y_train[idx]})')
    plt.axis('off')
plt.tight_layout()
plt.show()

# Предобработка данных
# Нормализация и преобразование в float32
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# One-hot encoding для меток
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(512, input_shape=(784,), activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(512, activation="relu"))# Добавление второго скрытого слоя
model.add(Dropout(0.2))

model.add(Dense(10, activation="softmax"))# Выходной полносвязный слой, 10 нейронов (по количеству классов)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

train = model.fit(x_train, y_train, batch_size=1024, epochs=40, validation_split=0.2, verbose=1)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

scores = model.evaluate(x_test, y_test, verbose=1)
print(f'Значение функции потерь (loss) на тестовых данных: {scores[0]}')
print(f'Доля верных ответов на тестовых данных, в процентах (accuracy): {round(scores[1] * 100, 4)}')

model.save("/content/fashion_model.keras")