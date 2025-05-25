import numpy as np

My_img = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1], dtype=np.uint8)
# My_img = np.array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
# My_img = np.array([1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
# My_img = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1], dtype=np.uint8)

POROG = 0.5
DATASET_CHOISE = "Large"  # Low, Medium, Large

DATASETS = {
    "Low": "Data/Storage/Datasets/Low/dataset.npz",
    "Medium": "Data/Storage/Datasets/Medium/dataset.npz",
    "Large": "Data/Storage/Datasets/Large/dataset.npz"
}

MY_DATASET = DATASETS[DATASET_CHOISE]

# Параметры нейросети
EPOCH = 200
LEARNING_RATE = 0.01
HIDDEN_NEURONS = 8

LOG = False
BUFFER_RANGE = 10
