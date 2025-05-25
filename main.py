import numpy as np
import time
import datetime
import os
from PIL import Image
from test import check_result
from Data.Storage.generate_dataset import generate_dataset
from Input.base_input_data import My_img, POROG, MY_DATASET, EPOCH, LEARNING_RATE, HIDDEN_NEURONS, LOG, BUFFER_RANGE
from Neurons.Train.forward_pass import forward_hidden_layer, forward_output_layer
from Neurons.Train.backpropagation import backpropagate
from Visual.visualization import visualize_neuron_weights

# Переменная для прекращения обучения при повторении BUFFER_RANGE раз
repeats = {}

print("="*100)
# Лучше сюда поместить генерацию датасета, чтобы я очередной инфаркт не ловил
for t in [1, 2, 3]:
    generate_dataset(dataset_type=t)

print(f"\nИспользуется датасет: {MY_DATASET}\n")
print("="*100)

# Загружаем датасет
data = np.load(MY_DATASET)
X = data['images']
y = data['labels']

input_size = My_img.shape[0]
output_size = 1

# Инициализация весов и биасов с улучшенной стратегией
# Xavier/Glorot инициализация
std_hidden = np.sqrt(2.0 / input_size)
std_output = np.sqrt(2.0 / HIDDEN_NEURONS)

W_hidden = np.random.normal(0, std_hidden, (HIDDEN_NEURONS, input_size))
W_output = np.random.normal(0, std_output, (output_size, HIDDEN_NEURONS))
bias_hidden = np.zeros(HIDDEN_NEURONS)
bias_output = np.zeros(output_size)

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
print(f"\nЭталонное изображение сохранено в {my_img_path}\n")

print("="*100)


# Функция для сохранения весов
def save_weights(W_hidden, W_output, bias_hidden, bias_output, epoch=None):
    weights_path = os.path.join(weights_dir, f"weights_final_{timestamp}.npz")
    np.savez(weights_path,
             W_hidden=W_hidden,
             W_output=W_output,
             bias_hidden=bias_hidden,
             bias_output=bias_output)
    print(f"Веса сохранены в {weights_path}")
    return weights_path


for epoch in range(EPOCH):
    correct = 0
    start = time.time()
    outputs_epoch = []
    predictions_1 = 0
    predictions_0 = 0

    for xi, target in zip(X, y):
        hidden_output = forward_hidden_layer(xi, W_hidden, bias_hidden)

        final_output = forward_output_layer(hidden_output, W_output, bias_output)

        outputs_epoch.append(final_output)

        if final_output > POROG:
            predicted = 1
            predictions_1 += 1
        else:
            predicted = 0
            predictions_0 += 1

        if predicted == target:
            correct += 1

        # Обратный проход
        W_hidden, W_output, bias_hidden, bias_output = backpropagate(
            xi, hidden_output, final_output, target,
            W_hidden, W_output,
            bias_hidden, bias_output,
            LEARNING_RATE
        )

    # Статистика
    acc = correct / len(X)
    acc_rounded = round(acc, 4)
    repeats[acc_rounded] = repeats.get(acc_rounded, 0) + 1

    if repeats[acc_rounded] >= BUFFER_RANGE:
        print(f"\nТочность {acc_rounded:.2%} повторяется {repeats[acc_rounded]} раз. Прерываем обучение.")
        break

    # Общий вывод
    print(f"\nЭпоха {epoch + 1}/{EPOCH}")
    print(f"Точность: {acc:.2%} | Правильных: {correct}/{len(X)}")
    print(f"Предсказания: класс 1 => {predictions_1}, класс 0 => {predictions_0}")
    print(f"Распределение в датасете: класс 1 => {sum(y)}, класс 0 => {len(y) - sum(y)}")

    if LOG:
        print("\nВеса от входов к скрытым нейронам:")
        for i in range(W_hidden.shape[0]):  # по скрытым нейронам
            row = ", ".join(f"{W_hidden[i, j]:+.4f}" for j in range(W_hidden.shape[1]))
            print(f"W_hid[{i}]: [{row}] | bias: {bias_hidden[i]:+.4f}")

        print("\nВеса от скрытых к выходному нейрону:")
        for i in range(W_output.shape[0]):  # по выходным нейронам
            row = ", ".join(f"{W_output[i, j]:+.4f}" for j in range(W_output.shape[1]))
            print(f"W_out[{i}]: [{row}] | bias: {bias_output[i]:+.4f}")

    print(f"\nВремя на эпоху: {time.time() - start:.2f}s")



    # Проверка на достижение высокой точности
    if acc >= 1:  # Снизил порог для более реалистичной цели
        print(f"Достигнута высокая точность ({acc:.2%}) на эпохе {epoch + 1}, обучение завершается.\n")
        weights_path = save_weights(W_hidden, W_output, bias_hidden, bias_output, epoch=epoch + 1)
        visualize_neuron_weights(W_hidden, visualizations_dir, epoch=epoch + 1)
        print("="*100)
        break

# Финальное сохранение и визуализация
if acc < 1:
    weights_path = save_weights(W_hidden, W_output, bias_hidden, bias_output)
    visualize_neuron_weights(W_hidden, visualizations_dir, epoch=epoch + 1)
    print(f"Обучение закончено с точностью {acc: .2%}\n")
    print("="*100)

check_result(weights_path)