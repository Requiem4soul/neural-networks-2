import os
import numpy as np
import datetime

def save_metadata(
    save_dir,
    My_img,
    POROG,
    DATASET_CHOISE,
    EPOCH,
    LAST_EPOCH,
    LEARNING_RATE,
    HIDDEN_NEURONS,
    BUFFER_RANGE,
    acc_final,
    total_time_sec
):
    metadata_path = os.path.join(save_dir, "metadata.txt")

    # Форматируем изображение в строки по 4 пикселя
    img_str_lines = []
    for i in range(0, len(My_img), 4):
        row = My_img[i:i+4]
        img_str_lines.append(" ".join(str(v) for v in row))

    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("Данные обучения\n")
        f.write("="*40 + "\n")
        f.write(f"Дата и время обучения: {datetime.datetime.now()}\n")
        f.write(f"Точность последней эпохи: {acc_final:.4f}\n")
        f.write(f"Время обучения: {total_time_sec:.2f} сек\n")
        f.write(f"Порог активационной функции: {POROG}\n")
        f.write(f"Выбранный датасет: {DATASET_CHOISE}\n")
        f.write(f"Макс. эпох: {EPOCH}\n")
        f.write(f"Итоговое количество эпох: {LAST_EPOCH}\n")
        f.write(f"Коэффициент скорости обучения: {LEARNING_RATE}\n")
        f.write(f"Количество скрытых нейронов: {HIDDEN_NEURONS}\n")
        f.write(f"Число окончания обучения, если точность не росла за N эпох: {BUFFER_RANGE}\n")
        f.write("\nИсходное изображение (4x4):\n")
        for line in img_str_lines:
            f.write(line + "\n")

    print(f"Метаданные сохранены в {metadata_path}")