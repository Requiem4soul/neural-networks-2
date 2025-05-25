import numpy as np
from Input.base_input_data import POROG, My_img
from Neurons.Train.forward_pass import forward_hidden_layer, forward_output_layer, sigmoid

# Генерация всех возможных изображений 4x4 (ИСПРАВЛЕНО)
def generate_all_images():
    total_images = 2 ** 16
    all_images = []

    # Генерируем все возможные 16-битные комбинации
    for i in range(total_images):
        # Преобразуем число в бинарное представление
        binary_str = format(i, '016b')  # 16-битное представление
        image = np.array([int(bit) for bit in binary_str], dtype=np.uint8)
        all_images.append(image)

    # Метки: 1 для My_img, 0 для всех остальных
    labels = [1 if np.array_equal(img, My_img) else 0 for img in all_images]

    print(f"Всего сгенерировано изображений: {len(all_images)}")
    print(f"Количество положительных примеров (My_img): {sum(labels)}")
    print(f"Эталонное изображение: {My_img}")

    return np.array(all_images), np.array(labels)

def check_result(final_weights):
    # Загружаем веса
    weights = np.load(final_weights)
    W_hidden = weights['W_hidden']
    W_output = weights['W_output']
    bias_hidden = weights['bias_hidden']
    bias_output = weights['bias_output']

    print("Загруженные веса:")
    print(f"W_hidden shape: {W_hidden.shape}")
    print(f"W_output shape: {W_output.shape}")
    print(f"bias_hidden shape: {bias_hidden.shape}")
    print(f"bias_output shape: {bias_output.shape}")

    # Генерируем все изображения
    X_test, y_test = generate_all_images()

    correct = 0
    predictions_1 = 0
    predictions_0 = 0
    false_positives = []  # Для анализа ложных срабатываний

    for i, (xi, target) in enumerate(zip(X_test, y_test)):
        # Прямой проход
        hidden_input = forward_hidden_layer(xi, W_hidden, bias_hidden)
        hidden_output = hidden_input
        final_input = forward_output_layer(hidden_output, W_output, bias_output)
        final_output = final_input.item()

        if final_output > POROG:
            predicted = 1
            predictions_1 += 1
            # Если предсказали 1, но это не эталонное изображение
            if target == 0:
                false_positives.append((i, xi, final_output))
        else:
            predicted = 0
            predictions_0 += 1

        if predicted == target:
            correct += 1

        # Показываем результат для эталонного изображения
        if target == 1:
            print(f"\nЭталонное изображение найдено на позиции {i}:")
            print(f"Изображение: {xi}")
            print(f"Выход сети: {final_output:.6f}")
            print(f"Предсказание: {predicted}")
            print(f"Правильно: {'Да' if predicted == target else 'Нет'}")

    # Вывод результатов
    acc = correct / len(X_test)
    print(f"\n{'=' * 50}")
    print(f"Точность распознования: {acc:.6f} ({acc:.2%}) | Правильно: {correct}/{len(X_test)}")
    print(f"Определено классов: 1 => {predictions_1}, 0 => {predictions_0}")
    print(f"Ошибок: {len(X_test) - correct}")

    # Анализ ложных срабатываний
    if false_positives:
        print(f"\nЛожные срабатывания:")
        print("Первые 5 ложных срабатываний:")
        for i, (pos, img, output) in enumerate(false_positives[:5]):
            print(f"  {i + 1}. Позиция {pos}: {img} (выход: {output:.6f})")

    if correct == 2 ** 16:
        print(f"\n0 Ошибок")
    else:
        print(f"\nНайдено {2 ** 16 - correct} ошибок из {2 ** 16} возможных")