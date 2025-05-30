import numpy as np
from Input.base_input_data import LOG

# Обратный проход или распространение обратной ошибки
# Мы первым шагом считаем ошибку, согласно формуле
# Далее, используя эту ошибку, мы подсчитываем градиент выходного нейрона
# Используя старые веса и градиент выходного нейрона, мы считаем насколько сильно влияли выходные веса скрытых нейронов
# В целом дальше идёт практически один в один, только уже от скрытых до входящих пикселей

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(sigmoid_output):
    return sigmoid_output * (1 - sigmoid_output)


def clip_weights(weights, threshold=5.0):
    """Ограничивает веса до заданного порога"""
    return np.clip(weights, -threshold, threshold)


def backpropagate(inputs, hidden_outputs, final_output, expected_output,
                  W_input_hidden, W_hidden_output,
                  bias_hidden, bias_output,
                  learning_rate):
    # 1. Ошибка выходного слоя (НЕ БЕРИ ТУТ АКТИВАЦИОННУЮ В СИГМОИДУ, У ТЕБЯ УЖЕ В ПРЯМОМ ПРОХОДЕ ОНА ИЗ СИГМОИДЫ)
    final_error = expected_output - final_output

    # 2. Градиент выходного слоя
    grad_output = final_error * sigmoid_derivative(final_output)

    for i in range(W_hidden_output.shape[1]):
        delta_W = learning_rate * grad_output * hidden_outputs[i]
        W_hidden_output[0, i] += delta_W

    # Обновление bias выходного слоя
    bias_output[0] += learning_rate * grad_output

    # 3. Обратная ошибка для скрытого слоя
    hidden_errors = np.zeros(len(hidden_outputs))

    for i in range(len(hidden_outputs)):
        hidden_errors[i] = grad_output * W_hidden_output[0, i] * sigmoid_derivative(hidden_outputs[i])

    # 4. Обновление весов скрытого слоя
    # W_input_hidden имеет форму (HIDDEN_NEURONS, input_size)
    for i in range(W_input_hidden.shape[0]):
        for j in range(W_input_hidden.shape[1]):
            delta_W = learning_rate * hidden_errors[i] * inputs[j]
            W_input_hidden[i, j] += delta_W

        # Обновление bias скрытого слоя
        bias_hidden[i] += learning_rate * hidden_errors[i]

    # Ограничиваем веса
    W_input_hidden = clip_weights(W_input_hidden, threshold=10.0)
    W_hidden_output = clip_weights(W_hidden_output, threshold=10.0)

    return W_input_hidden, W_hidden_output, bias_hidden, bias_output