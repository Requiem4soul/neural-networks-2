import numpy as np
import time
import datetime
import os
from PIL import Image
from Input.base_input_data import My_img, POROG, MY_DATASET, EPOCH, LEARNING_RATE, HIDDEN_NEURONS
from Neurons.Train.forward_pass import forward_hidden_layer, forward_output_layer, sigmoid
from Neurons.Train.backpropagation import backpropagate
from Visual.visualization import visualize_neuron_weights

# Загружаем датасет
data = np.load(MY_DATASET)
X = data['images']
y = data['labels']

input_size = My_img.shape[0]
output_size = 1

# Инициализация весов и биасов
W_hidden = np.random.uniform(-0.1, 0.1, (HIDDEN_NEURONS, input_size))
W_output = np.random.uniform(-0.1, 0.1, (output_size, HIDDEN_NEURONS))
bias_hidden = np.zeros(HIDDEN_NEURONS)
bias_output = np.zeros(output_size)

print(">>> INITIAL WEIGHT RANGES <<<")
print(f"W_hidden: {W_hidden.min():.4f} to {W_hidden.max():.4f}")
print(f"W_output: {W_output.min():.4f} to {W_output.max():.4f}\n")

# Создаём папку для сохранения с временной меткой
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
save_base_dir = os.path.join("Data", "Storage", "Saves", f"training_{timestamp}")
os.makedirs(save_base_dir, exist_ok=True)
weights_dir = os.path.join(save_base_dir, "weights")
visualizations_dir = os.path.join(save_base_dir, "neuron_visualizations")
os.makedirs(weights_dir, exist_ok=True)
os.makedirs(visualizations_dir, exist_ok=True)

# Сохраняем My_img как PNG
my_img_array = My_img.reshape(4, 4) * 255  # 0->0 (чёрный), 1->255 (белый)
my_img = Image.fromarray(my_img_array.astype(np.uint8), mode='L')
my_img_path = os.path.join(save_base_dir, "target_image.png")
my_img.save(my_img_path)
print(f"Target image saved to {my_img_path}")

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
    hidden_vals = []

    for xi, target in zip(X, y):
        # --- Прямой проход ---
        hidden_input = forward_hidden_layer(xi, W_hidden, bias_hidden)
        hidden_output = sigmoid(hidden_input)
        hidden_vals.append(hidden_output)

        final_input = forward_output_layer(hidden_output, W_output, bias_output)
        final_output = sigmoid(final_input).item()

        outputs_epoch.append(final_output)

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
    print(f"\nEpoch {epoch+1}/{EPOCH}")
    print(f"Accuracy     : {acc:.2%} | Correct: {correct}/{len(X)}")
    print(f"Predictions  : 1 => {predictions_1}, 0 => {predictions_0}")
    print(f"Weights range: W_hid={W_hidden.min():.4f}..{W_hidden.max():.4f} | W_out={W_output.min():.4f}..{W_output.max():.4f}")
    print(f"Time taken   : {time.time() - start:.2f}s")

    # Проверка на достижение 100% точности
    if acc >= 1.0:
        print(f"Reached 100% accuracy at epoch {epoch+1}, stopping training.")
        save_weights(W_hidden, W_output, bias_hidden, bias_output)
        visualize_neuron_weights(W_hidden, visualizations_dir, epoch=epoch+1)
        break

# Финальное сохранение и визуализация
if acc < 1.0:
    save_weights(W_hidden, W_output, bias_hidden, bias_output)
    visualize_neuron_weights(W_hidden, visualizations_dir, epoch=epoch+1)