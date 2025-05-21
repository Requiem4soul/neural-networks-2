import numpy as np
import time
import datetime
from Input.base_input_data import My_img, POROG, MY_DATASET, EPOCH, LEARNING_RATE, HIDDEN_NEURONS
from Neurons.Train.forward_pass import forward_hidden_layer, forward_output_layer, sigmoid
from Neurons.Train.backpropagation import backpropagate

# Загружаем датасет
data = np.load(MY_DATASET)
X = data['images']
y = data['labels']

input_size = My_img.shape[0]
output_size = 1

# Инициализация весов и биасов
# Инициализация весов и биасов случайными значениями из диапазона [-0.1, 0.1]
W_hidden = np.random.uniform(-0.1, 0.1, (HIDDEN_NEURONS, input_size))
W_output = np.random.uniform(-0.1, 0.1, (output_size, HIDDEN_NEURONS))

bias_hidden = np.zeros(HIDDEN_NEURONS)
bias_output = np.zeros(output_size)

print(">>> INITIAL WEIGHT RANGES <<<")
print(f"W_hidden: {W_hidden.min():.4f} to {W_hidden.max():.4f}")
print(f"W_output: {W_output.min():.4f} to {W_output.max():.4f}\n")

# Функция для сохранения весов
def save_weights(W_hidden, W_output, bias_hidden, bias_output, epoch=None):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if epoch is not None:
        weights_path = f"weights_epoch_{epoch}_{timestamp}.npz"
    else:
        weights_path = f"weights_final_{timestamp}.npz"
    np.savez(weights_path,
             W_hidden=W_hidden,
             W_output=W_output,
             bias_hidden=bias_hidden,
             bias_output=bias_output)
    print(f"Weights saved to {weights_path}")

for epoch in range(EPOCH):
    correct = 0
    start = time.time()

    outputs_epoch = []
    predictions_1 = 0
    predictions_0 = 0
    loss_epoch = 0.0
    hidden_vals = []

    for xi, target in zip(X, y):
        # --- Прямой проход ---
        hidden_input = forward_hidden_layer(xi, W_hidden, bias_hidden)
        hidden_output = sigmoid(hidden_input)
        hidden_vals.append(hidden_output)

        final_input = forward_output_layer(hidden_output, W_output, bias_output)
        final_output = sigmoid(final_input).item()

        outputs_epoch.append(final_output)
        # loss_epoch += 0.5 * (final_output - target) ** 2  # MSE

        if final_output > POROG:
            predicted = 1
        else:
            predicted = 0

        if predicted == 1:
            predictions_1 += 1
        else:
            predictions_0 += 1

        if predicted == target:
            correct += 1

        # --- Обратный проход ---
        W_hidden, W_output, bias_hidden, bias_output = backpropagate(
            xi, hidden_output, final_output, target,
            W_hidden, W_output,
            bias_hidden, bias_output,
            LEARNING_RATE
        )

    # Статистика
    acc = correct / len(X)
    # loss_epoch /= len(X)
    # hidden_vals = np.array(hidden_vals)

    print(f"\nEpoch {epoch+1}/{EPOCH}")
    print(f"Accuracy     : {acc:.2%} | Correct: {correct}/{len(X)}")
    print(f"Predictions  : 1 => {predictions_1}, 0 => {predictions_0}")
    print(f"Weights range: W_hid={W_hidden.min():.4f}..{W_hidden.max():.4f} | W_out={W_output.min():.4f}..{W_output.max():.4f}")
    print(f"Time taken   : {time.time() - start:.2f}s")

    # Сохранение весов каждые 10 эпох
    if (epoch + 1) % 10 == 0:
        save_weights(W_hidden, W_output, bias_hidden, bias_output, epoch=epoch+1)

    # Проверка на достижение 100% точности
    if acc >= 1.0:
        print(f"Reached 100% accuracy at epoch {epoch+1}, stopping training.")
        save_weights(W_hidden, W_output, bias_hidden, bias_output)  # Финальное сохранение
        break

# Финальное сохранение, если обучение завершилось без достижения 100% точности
if acc < 1.0:
    save_weights(W_hidden, W_output, bias_hidden, bias_output)