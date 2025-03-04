import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import LipFrameDataset
from model import LipDeepfakeDetector
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
data = np.load('data/lip_data_checkpoint.npz')
all_lip_frames = data['lip_frames']
labels = data['labels'].tolist()
train_frames, test_frames, train_labels, test_labels = train_test_split(
    all_lip_frames, labels, test_size=0.2, random_state=42
)
train_dataset = LipFrameDataset(train_frames, train_labels)
test_dataset = LipFrameDataset(test_frames, test_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_frames = 10
model = LipDeepfakeDetector(max_frames=max_frames).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
model_checkpoint = f'models/cnn_gru_{max_frames}_frames.pth'

# Early stopping
patience = 5
early_stop_counter = 0
best_accuracy = 0
best_epoch = 0
train_acc_history = []
val_acc_history = []
epochs = 40

# Training
for epoch in range(epochs):
    model.train()
    total_loss = 0
    train_correct = 0
    train_total = 0
    for lip_frames, lbls in train_loader:
        lip_frames, lbls = lip_frames.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = model(lip_frames)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += lbls.size(0)
        train_correct += (predicted == lbls).sum().item()
    train_loss = total_loss / len(train_loader)
    train_accuracy = 100 * train_correct / train_total
    train_acc_history.append(train_accuracy)

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for lip_frames, lbls in test_loader:
            lip_frames, lbls = lip_frames.to(device), lbls.to(device)
            outputs = model(lip_frames)
            _, predicted = torch.max(outputs, 1)
            val_total += lbls.size(0)
            val_correct += (predicted == lbls).sum().item()
    val_accuracy = 100 * val_correct / val_total
    val_acc_history.append(val_accuracy)

    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%')
    scheduler.step(train_loss)
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_epoch = epoch
        early_stop_counter = 0
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'accuracy': val_accuracy}, model_checkpoint)
        print(f'New best checkpoint saved at epoch {epoch+1} with Val Accuracy {val_accuracy:.2f}%')
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break

# Plot Accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_acc_history) + 1), train_acc_history, label='Train Accuracy', marker='o')
plt.plot(range(1, len(val_acc_history) + 1), val_acc_history, label='Val Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Train Accuracy vs Val Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

print('Training completed!')