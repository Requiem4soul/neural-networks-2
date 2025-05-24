import numpy as np

# Путь к файлу с весами
weights_path = "Data/Storage/Saves/training_20250525_002601/weights/weights_final_20250525_002601.npz"

# Загружаем веса
data = np.load(weights_path)
W_input_hidden = data['W_hidden']
W_hidden_output = data['W_output']
bias_hidden = data['bias_hidden']
bias_output = data['bias_output']

# Проверяем форму
print(f"W_input_hidden shape: {W_input_hidden.shape}")
print(f"W_hidden_output shape: {W_hidden_output.shape}")
print(f"bias_hidden shape: {bias_hidden.shape}")
print(f"bias_output shape: {bias_output.shape}")

# Анализируем W_input_hidden (веса для скрытых нейронов)
print("\nW_input_hidden analysis:")
num_neurons = W_input_hidden.shape[0]
for i in range(num_neurons):
    weights = W_input_hidden[i]
    print(f"Neuron {i}: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}, std={weights.std():.4f}")
    print(f"  Weights (first 5): {weights[:5]}")

# Проверяем уникальность нейронов
print("\nChecking neuron weight similarity:")
for i in range(num_neurons):
    for j in range(i + 1, num_neurons):
        diff = np.abs(W_input_hidden[i] - W_input_hidden[j]).mean()
        if diff < 0.1:  # Порог для "похожести"
            print(f"Neurons {i} and {j} are similar (mean diff={diff:.4f})")