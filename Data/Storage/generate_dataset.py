import numpy as np
import os
import shutil
from Input.base_input_data import My_img
import random


def generate_dataset(dataset_type=1):
    """
    Генерирует датасет с чёрно-белыми изображениями 4x4.
    Баланс: 90% ложных, 10% истинных примеров.
    """
    config = {
        1: {"path": "Datasets/Low", "rate": 0.97},
        2: {"path": "Datasets/Medium", "rate": 0.95},
        3: {"path": "Datasets/Large", "rate": 0.93}
    }

    if dataset_type not in config:
        raise ValueError("Доступные типы датасетов: 1 (Low), 2 (Medium), 3 (Large)")

    rate = config[dataset_type]["rate"]
    dataset_path = config[dataset_type]["path"]

    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    os.makedirs(dataset_path, exist_ok=True)

    negative_images = []
    total_images = 2 ** 16
    current_image = np.zeros(16, dtype=np.uint8)

    for _ in range(total_images):
        if random.random() > rate:
            if not np.array_equal(current_image, My_img):
                negative_images.append(current_image.copy())

        # Инкремент в бинарном формате
        for i in range(16):
            current_image[i] ^= 1
            if current_image[i] == 1:
                break

    # Вычисляем нужное количество положительных примеров
    n_negative = len(negative_images)
    n_positive = int(n_negative / 0.5)  # 10% от общего => pos = neg / 9

    positive_images = [My_img.copy() for _ in range(n_positive)]
    positive_labels = [1] * n_positive
    negative_labels = [0] * n_negative

    # Объединяем и перемешиваем
    all_images = negative_images + positive_images
    all_labels = negative_labels + positive_labels

    combined = list(zip(all_images, all_labels))
    random.shuffle(combined)

    shuffled_images, shuffled_labels = zip(*combined)

    # Сохраняем
    np.savez(
        os.path.join(dataset_path, "dataset.npz"),
        images=np.array(shuffled_images),
        labels=np.array(shuffled_labels)
    )

    print("═" * 50)
    print(f"Датасет пересоздан в {dataset_path}")
    print(f"Всего изображений: {len(all_images)}")
    print(f"Из них эталонных: {sum(shuffled_labels)}")
    print("═" * 50 + "\n")


if __name__ == "__main__":
    for t in [1, 2, 3]:
        generate_dataset(dataset_type=t)
