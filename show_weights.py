import numpy as np
import os

file_path = "Data/Storage/Saves/training_20250525_225237/weights/weights_final_20250525_225237.npz"

data = np.load(file_path)
W_hidden = data["W_hidden"]
W_output = data["W_output"]
bias_hidden = data["bias_hidden"]
bias_output = data["bias_output"]

# Вывод весов от входов к скрытым нейронам
print("\nВеса от входов к скрытым нейронам:")
for i in range(W_hidden.shape[0]):  # по скрытым нейронам
    row = ", ".join(f"{W_hidden[i, j]:+.4f}" for j in range(W_hidden.shape[1]))
    print(f"W_hid[{i}]: [{row}] | bias: {bias_hidden[i]:+.4f}")

# Вывод весов от скрытых к выходному нейрону
print("\nВеса от скрытых к выходному нейрону:")
for i in range(W_output.shape[0]):  # по выходным нейронам
    row = ", ".join(f"{W_output[i, j]:+.4f}" for j in range(W_output.shape[1]))
    print(f"W_out[{i}]: [{row}] | bias: {bias_output[i]:+.4f}")
