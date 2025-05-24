import numpy as np
from PIL import Image
import os

def visualize_neuron_weights(W_input_hidden, save_dir, epoch):
    """
    Создаёт тепловые карты 4x4 для каждого скрытого нейрона на основе весов W_input_hidden.
    W_input_hidden: матрица [HIDDEN_NEURONS, input_size], где input_size=16.
    save_dir: папка для сохранения изображений.
    epoch: номер эпохи для имени файла.
    """
    os.makedirs(save_dir, exist_ok=True)
    num_neurons = W_input_hidden.shape[0]  # HIDDEN_NEURONS
    input_size = W_input_hidden.shape[1]  # 16 пикселей

    for neuron_idx in range(num_neurons):
        # Получаем веса для нейрона i (вектор длиной 16)
        weights = W_input_hidden[neuron_idx]

        # Фиксированная нормализация: [-5, 5] -> [0, 255]
        normalized_weights = (weights + 5) / 10 * 255
        normalized_weights = np.clip(normalized_weights, 0, 255).astype(np.uint8)

        # Преобразуем вектор 16 в матрицу 4x4
        weight_image = normalized_weights.reshape(4, 4)

        # Создаём изображение
        img = Image.fromarray(weight_image, mode='L')  # 'L' — градации серого

        # Сохраняем
        img_path = os.path.join(save_dir, f"neuron_{neuron_idx}_epoch_{epoch}.png")
        img.save(img_path)