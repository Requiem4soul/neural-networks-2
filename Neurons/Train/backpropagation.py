import numpy as np
from Input.base_input_data import LOG


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
    # 1. Ошибка выходного слоя
    final_error = expected_output - final_output

    if LOG:
        print("\n[ОШИБКА]")
        print(f"Ожидаемый выход: {expected_output}")
        print(f"Фактический выход: {final_output:.4f}")
        print(f"Вычисленная ошибка: {final_error:.4f}")
        print(f"Величина обновления: {final_error * sigmoid_derivative(final_output):.4f}")

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
    for i in range(W_input_hidden.shape[0]):  # по скрытым нейронам
        for j in range(W_input_hidden.shape[1]):  # по входным признакам
            delta_W = learning_rate * hidden_errors[i] * inputs[j]
            W_input_hidden[i, j] += delta_W

        # Обновление bias скрытого слоя
        bias_hidden[i] += learning_rate * hidden_errors[i]

    # Ограничиваем веса
    W_input_hidden = clip_weights(W_input_hidden, threshold=10.0)
    W_hidden_output = clip_weights(W_hidden_output, threshold=10.0)

    return W_input_hidden, W_hidden_output, bias_hidden, bias_output