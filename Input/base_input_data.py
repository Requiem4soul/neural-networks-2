import numpy as np

# Тут хранится единственно правильное изображение в 4 на 4 пикселя
# My_img = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1], dtype=np.uint8) # Сложность высокая
# My_img = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8) # Средний-соложный вариант 2
My_img = np.array([1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1], dtype=np.uint8) # Средний-соложный вариант 1
# My_img = np.array([1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1], dtype=np.uint8) # Средний-лёгкий вариант
# My_img = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.uint8) # Самый лёгкий вариант и самый быстрый


# Задаём пороговую функцию
POROG = 0.6

# Выбор датасета
DATASET_CHOISE = "Medium" # "Low", "Medium", "Large"

DATASETS = {
    "Low": "Data/Storage/Datasets/Low/dataset.npz",
    "Medium": "Data/Storage/Datasets/Medium/dataset.npz",
    "Large": "Data/Storage/Datasets/Large/dataset.npz"
}

MY_DATASET = DATASETS[DATASET_CHOISE]

# Параметры сети
EPOCH = 300
LEARNING_RATE = 0.005
HIDDEN_NEURONS = 8

# Параметры иные
LOG = False
