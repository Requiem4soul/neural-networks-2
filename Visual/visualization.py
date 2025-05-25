import numpy as np
from PIL import Image
import os


def visualize_neuron_weights(W_input_hidden, save_dir, epoch):
    """
    Создаёт цветные тепловые карты 4x4 для каждого скрытого нейрона.
    Цветовая схема:
        - [-1, 0] → чёрный-серый (R=G=B: 0...128)
        - (0, 1] → серый-красный (R: 128...255, G=B: 128...0)
    """
    os.makedirs(save_dir, exist_ok=True)
    num_neurons = W_input_hidden.shape[0]  # HIDDEN_NEURONS
    input_size = W_input_hidden.shape[1]  # 16 пикселей

    for neuron_idx in range(num_neurons):
        weights = W_input_hidden[neuron_idx]

        # Создаём RGB-матрицу 4x4x3
        rgb_image = np.zeros((4, 4, 3), dtype=np.uint8)

        for i, weight in enumerate(weights):
            row, col = i // 4, i % 4

            if weight <= 0:  # Диапазон [-1, 0]
                intensity = int(128 * (1 + max(weight, -1)))  # 0...128
                rgb_image[row, col] = [intensity, intensity, intensity]
            else:  # Диапазон (0, 1]
                red = 128 + int(127 * min(weight, 1))
                green_blue = 128 - int(128 * min(weight, 1))
                rgb_image[row, col] = [red, green_blue, green_blue]

        # Создаём и сохраняем изображение
        img = Image.fromarray(rgb_image, mode='RGB')
        img_path = os.path.join(save_dir, f"neuron_{neuron_idx}_epoch_{epoch}.png")
        img.save(img_path)