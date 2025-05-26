import numpy as np

# Обычный прямой проход с тем отличием, что нужно учитывать что ещё нужно применять активационную функцию
# В остальном схоже по работе с перцептроном

def sigmoid(x):
    x = np.clip(x, -500, 500)  # Ограничение входных значений
    return 1 / (1 + np.exp(-x))

def forward_hidden_layer(inputs, weights, bias):
    """
    Прямой проход через скрытый слой.
    """
    net = np.dot(weights, inputs) + bias
    return sigmoid(net)

def forward_output_layer(hidden_activations, weights, bias):
    """
    Прямой проход через выходной слой.
    weights: вектор весов от каждого скрытого нейрона к выходному нейрону
    """
    net = np.dot(weights, hidden_activations) + bias
    return sigmoid(net)  # активация выходного нейрона

def predict(output_activation, POROG=0.4):
    """
    Смотрим активировался ли выходной нейрон
    """
    return output_activation > POROG