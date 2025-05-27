import numpy as np
import os
def print_weights(W_hidden, W_output, bias_hidden, bias_output):

    print("\nВеса от входов к скрытым нейронам:")
    for i in range(W_hidden.shape[0]):
        row = ", ".join(f"{W_hidden[i, j]:+.4f}" for j in range(W_hidden.shape[1]))
        print(f"W_hid[{i}]: [{row}] | bias: {bias_hidden[i]:+.4f}")

    print("\nВеса от скрытых к выходному нейрону:")
    for i in range(W_output.shape[0]):
        row = ", ".join(f"{W_output[i, j]:+.4f}" for j in range(W_output.shape[1]))
        print(f"W_out[{i}]: [{row}] | bias: {bias_output[i]:+.4f}")
