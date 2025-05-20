import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from dataset import EmotionDataset, load_metadata, wrapped_transform
from model import CNNEmotionClassifier, train_model, plot_metrics
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    audio_dir = "ravdess/archive/audio_speech_actors_01-24"
    model_path = "emotion_cnn.pth"

    df = load_metadata(audio_dir)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['emotion_label'], random_state=42)

    train_dataset = EmotionDataset(train_df, transform=wrapped_transform)
    val_dataset = EmotionDataset(val_df, transform=wrapped_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNEmotionClassifier(n_classes=8).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("\n>>> Загружена предобученная модель.")
    else:
        print("\n>>> Обучение модели с нуля...")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        train_losses, val_losses, train_acc, val_acc = train_model(
            model, train_loader, val_loader,
            criterion, optimizer, device,
            epochs=1000,
            scheduler=scheduler,
            patience=150,
            model_path=model_path
        )

        torch.save(model.state_dict(), model_path)
        print("\n>>> Модель сохранена в", model_path)
        plot_metrics(train_losses, val_losses, train_acc, val_acc)

    # Классификация первого примера из валидационного сета
    model.eval()
    x, label = val_dataset[0]
    with torch.no_grad():
        output = model(x.unsqueeze(0).to(device))
        predicted = torch.argmax(output, dim=1).item()

    reverse_label_map = {v: k for k, v in train_dataset.label_map.items()}
    print(f"\n>>> Истинная эмоция: {reverse_label_map[label]}, Предсказанная: {reverse_label_map[predicted]}")

    # Вычислим предсказания для всей validation выборки
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    # Построим confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    labels = [reverse_label_map[i] for i in range(8)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Предсказано")
    plt.ylabel("Истинное значение")
    plt.title("Матрица ошибок (Confusion Matrix)")
    plt.tight_layout()
    plt.show()