import numpy as np


My_img = np.array([1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1], dtype=np.uint8)

# Настройки для более сложного обучения
POROG = 0.5
DATASET_CHOISE = "Medium"  # Используем больший датасет

DATASETS = {
    "Low": "Data/Storage/Datasets/Low/dataset.npz",
    "Medium": "Data/Storage/Datasets/Medium/dataset.npz",
    "Large": "Data/Storage/Datasets/Large/dataset.npz"
}

MY_DATASET = DATASETS[DATASET_CHOISE]

# Параметры для более медленного, но стабильного обучения
EPOCH = 50
LEARNING_RATE = 0.01  # Меньше learning rate
HIDDEN_NEURONS = 8    # Меньше нейронов = сложнее задача

LOG = False