import numpy as np
import pickle  # Импортируем библиотеку pickle

# Функция активации (сигмоида)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# сумма входных данных подаётся в сигмоиду, а она формирует выход нейрона f11
# Производная сигмоиды
def sigmoid_derivative(x):
    return x * (1 - x)

# Класс нейрона
class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
# Принимает входные данные, перемножает их на веса, добавляет смещение и пропускает через функцию активации (например, сигмоиду).
# Результат — выход нейрона (число в диапазоне от 0 до 1).
    def feedforward(self, inputs):
        total = np.dot(inputs, self.weights) + self.bias # v11 - Сумма входных данных
        return sigmoid(total)

# Класс простой нейронной сети с одним скрытым слоем
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size): #self, скрытые слои,
        # Создаем скрытый слой с заданным количеством нейронов
        self.hidden_layer = [Neuron(input_size) for _ in range(hidden_size)] #создаём список из множества нейронов
        # Создаем выходной слой с заданным количеством нейронов
        self.output_layer = [Neuron(hidden_size) for _ in range(output_size)]

    def feedforward(self, inputs):
        # Прямое распространение через скрытый слой
        hidden_outputs = [neuron.feedforward(inputs) for neuron in self.hidden_layer] # формируем список выходов нейрона f11 и т.д в нашем первом слое
        # Прямое распространение через выходной слой
        outputs = [neuron.feedforward(hidden_outputs) for neuron in self.output_layer]
        return outputs #список float
# train - метод для обучения нейрона. На основе входных данных и целевых меток корректирует веса и смещение с помощью градиентного спуска.
    def train(self, inputs, targets, learning_rate, epochs):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                # Прямое распространение через скрытый слой
                hidden_outputs = [neuron.feedforward(inputs[i]) for neuron in self.hidden_layer]
                # Прямое распространение через выходной слой
                outputs = [neuron.feedforward(hidden_outputs) for neuron in self.output_layer]

                # Ошибка на выходном слое
                errors = [targets[i][j] - outputs[j] for j in range(len(outputs))]

                # Обратное распространение ошибки на выходной слой
                d_outputs = [errors[j] * sigmoid_derivative(outputs[j]) for j in range(len(outputs))]
                for j, neuron in enumerate(self.output_layer):
                    neuron.weights += learning_rate * d_outputs[j] * np.array(hidden_outputs)
                    neuron.bias += learning_rate * d_outputs[j]

                # Обратное распространение ошибки на скрытый слой
                d_hidden = [sum(d_outputs[j] * self.output_layer[j].weights[k] for j in range(len(outputs))) *
                             sigmoid_derivative(hidden_outputs[k]) for k in range(len(self.hidden_layer))]
                for k, neuron in enumerate(self.hidden_layer):
                    neuron.weights += learning_rate * d_hidden[k] * inputs[i]
                    neuron.bias += learning_rate * d_hidden[k]

            # Логирование процесса обучения
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}")

    def save(self, filename):
        """Сохраняет модель в файл с использованием pickle."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Загружает модель из файла с использованием pickle."""
        with open(filename, 'rb') as f:
            return pickle.load(f)