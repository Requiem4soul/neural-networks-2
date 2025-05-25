import numpy as np
from Input.base_input_data import My_img, POROG
from Neurons.Train.forward_pass import forward_hidden_layer, forward_output_layer, sigmoid


# Генерация всех возможных изображений 4x4
def generate_all_images():
    total_images = 2 ** 16
    all_images = []
    current_image = np.zeros(16, dtype=np.uint8)

    for _ in range(total_images):
        all_images.append(current_image.copy())
        # Инкремент в бинарном формате
        for i in range(16):
            current_image[i] ^= 1
            if current_image[i] == 1:
                break

    # Метки: 1 для My_img, 0 для всех остальных
    labels = [1 if np.array_equal(img, My_img) else 0 for img in all_images]
    return np.array(all_images), np.array(labels)


# Загружаем веса
weights = np.load("Data/Storage/Saves/training_20250525_133656/weights/weights_final_20250525_133656.npz")  # Укажите путь к нужному файлу весов
W_hidden = weights['W_hidden']
W_output = weights['W_output']
bias_hidden = weights['bias_hidden']
bias_output = weights['bias_output']

# Генерируем все изображения
X_test, y_test = generate_all_images()

correct = 0
predictions_1 = 0
predictions_0 = 0

for xi, target in zip(X_test, y_test):
    # Прямой проход
    hidden_input = forward_hidden_layer(xi, W_hidden, bias_hidden)
    hidden_output = sigmoid(hidden_input)
    final_input = forward_output_layer(hidden_output, W_output, bias_output)
    final_output = sigmoid(final_input).item()

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

# Вывод результатов
acc = correct / len(X_test)
print(f"Test Accuracy: {acc:.2%} | Correct: {correct}/{len(X_test)}")
print(f"Test Predictions: 1 => {predictions_1}, 0 => {predictions_0}\n")

if correct == 2**16:
    print(f"0 Ошибок - это идеальный результат")