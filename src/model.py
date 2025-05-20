import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class CNNEmotionClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        
        # Сверточные слои
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  # [B, 1, 32, 800] → [B, 16, 16, 400]
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # → [B, 16, 8, 200]

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # [B, 16, 8, 200] → [B, 32, 8, 200]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                           # → [B, 32, 1, 1]
        )

        # Полносвязный слой
        self.fc = nn.Sequential(
            nn.Flatten(),              # [B, 32, 1, 1] → [B, 32]
            nn.Linear(32, n_classes)   # → [B, 8]
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=1000, scheduler=None, patience=30, model_path="best_model.pth"):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    best_val_loss = float('inf')
    trigger_times = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct_train = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == y).sum().item()

        train_losses.append(running_loss / len(train_loader))
        train_acc = correct_train / len(train_loader.dataset)
        train_accuracies.append(train_acc)

        # Валидация
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                val_loss += criterion(outputs, y).item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y).sum().item()
                
        val_loss_avg = val_loss / len(val_loader)
        val_losses.append(val_loss_avg)
        acc = correct / len(val_loader.dataset)
        val_accuracies.append(acc)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss_avg:.4f}, Acc: {acc:.4f}")

        # Снижение learning rate
        if scheduler:
            scheduler.step(val_loss_avg)

        # Проверка на улучшение
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            trigger_times = 0
            torch.save(model.state_dict(), model_path)
            print(f"→ New best model saved at epoch {epoch+1} (val_loss={val_loss_avg:.4f})")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    return train_losses, val_losses, train_accuracies, val_accuracies


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Потери
    ax[0].plot(epochs, train_losses, label='Потери на обучении')
    ax[0].plot(epochs, val_losses, label='Потери на валидации')
    ax[0].set_title('Функция потерь')
    ax[0].set_xlabel('Эпоха')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].grid(True)

    # Точность
    ax[1].plot(epochs, train_accuracies, label='Точность на обучении')
    ax[1].plot(epochs, val_accuracies, label='Точность на валидации')
    ax[1].set_title('Точность модели')
    ax[1].set_xlabel('Эпоха')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()
