import numpy as np
from Input.base_input_data import LOG

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sigmoid_output):
    return sigmoid_output * (1 - sigmoid_output)

def backpropagate(inputs, hidden_outputs, final_output, expected_output,
                  W_input_hidden, W_hidden_output,
                  bias_hidden, bias_output,
                  learning_rate):

    # 1. Ошибка выходного слоя
    output_error = expected_output - final_output
    if LOG:
        print(f"[BP] Output error: {output_error:.6f}")

    # 2. Градиент выхода
    output_gradient = sigmoid_derivative(final_output) * output_error
    if LOG:
        print(f"[BP] Output gradient: {output_gradient:.6f}")

    # 3. Обновление весов и биаса выходного слоя
    for i in range(len(W_hidden_output[0])):  # HIDDEN_NEURONS
        delta = learning_rate * output_gradient * hidden_outputs[i]
        W_hidden_output[0][i] += delta
        if LOG:
            print(f"[BP] ΔW_out[{i}]: {delta:.6f}")
    bias_delta = learning_rate * output_gradient
    bias_output[0] += bias_delta
    if LOG:
        print(f"[BP] ΔBias_output: {bias_delta:.6f}")

    # 4. Обратная ошибка скрытого слоя
    for i in range(len(hidden_outputs)):  # скрытые нейроны
        hidden_error = output_gradient * W_hidden_output[0][i]
        hidden_gradient = sigmoid_derivative(hidden_outputs[i]) * hidden_error
        hidden_gradient = np.clip(hidden_gradient, -1.0, 1.0)  # Клиппинг градиента
        for j in range(len(inputs)):  # входы
            delta = learning_rate * hidden_gradient * inputs[j]
            W_input_hidden[i][j] += delta
            if LOG:
                print(f"[BP] ΔW_hid[{i}][{j}]: {delta:.6f}")
        bias_hid_delta = learning_rate * hidden_gradient
        bias_hidden[i] += bias_hid_delta
        if LOG:
            print(f"[BP] ΔBias_hidden[{i}]: {bias_hid_delta:.6f}")

    # 5. Клипим веса после обновления
    W_input_hidden = np.clip(W_input_hidden, -10, 10)
    W_hidden_output = np.clip(W_hidden_output, -10, 10)

    if LOG:
        print(f"[BP] W_input_hidden range: {np.min(W_input_hidden):.4f} .. {np.max(W_input_hidden):.4f}")
        print(f"[BP] W_hidden_output range: {np.min(W_hidden_output):.4f} .. {np.max(W_hidden_output):.4f}")

    return W_input_hidden, W_hidden_output, bias_hidden, bias_output
