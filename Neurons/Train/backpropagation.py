import numpy as np
from Input.base_input_data import LOG

def sigmoid(x):
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
    final_error =  expected_output - final_output

    # 2. Градиент выхода
    grad_output = final_error * sigmoid_derivative(final_output)
    for i in range(len(W_hidden_output[0])):
        delta_W_hidden_output = learning_rate * grad_output * hidden_outputs[i]
        W_hidden_output[0][i] = W_hidden_output[0][i] + delta_W_hidden_output

    delta_bias_output = learning_rate * grad_output
    bias_output[0] = bias_output[0] + delta_bias_output


    # 3. Обратная ошибка скрытого слоя
    hidden_error = np.zeros(32)

    for i in range(len(hidden_error)):
        hidden_error[i] = grad_output * W_hidden_output[0][i] * sigmoid_derivative(hidden_outputs[i])

    # 4. Обновление весов и смещения скрытого слоя
    for i in range(len(W_input_hidden)):
        for j in range(len(inputs)):
            delta_W_input = learning_rate * hidden_error[i] * inputs[j]
            W_input_hidden[i][j] = W_input_hidden[i][j] + delta_W_input

        delta_bias_input = learning_rate * hidden_error[i]
        bias_hidden[i] = bias_hidden[i] + delta_bias_input

    # Ограничиваем веса для предотвращения их взрывного роста
    W_input_hidden = clip_weights(W_input_hidden, threshold=5.0)
    W_hidden_output = clip_weights(W_hidden_output, threshold=5.0)

    return W_input_hidden, W_hidden_output, bias_hidden, bias_output
