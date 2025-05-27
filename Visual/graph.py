import matplotlib.pyplot as plt
import os


def save_accuracy_graph(epoch_stat, save_base_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(epoch_stat) + 1), epoch_stat, marker='o', linestyle='-')
    plt.title('Точность по эпохам')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.grid(True)
    plt.tight_layout()

    # Сохраняем график
    graph_path = os.path.join(save_base_dir, "training_accuracy.png")
    plt.savefig(graph_path)
    plt.close()
    print(f"График точности сохранён в {graph_path}")