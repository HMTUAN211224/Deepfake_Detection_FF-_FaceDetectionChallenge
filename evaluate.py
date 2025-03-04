import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import LipFrameDataset
from model import LipDeepfakeDetector
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Evaluate Deepfake Lip Detection Model')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
args = parser.parse_args()

# Load data
data = np.load('data/lip_data_checkpoint.npz')
all_lip_frames = data['lip_frames']
labels = data['labels'].tolist()
_, test_frames, _, test_labels = train_test_split(
    all_lip_frames, labels, test_size=0.2, random_state=42
)
test_dataset = LipFrameDataset(test_frames, test_labels)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LipDeepfakeDetector(max_frames=10).to(device)
checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Evaluate
test_correct = 0
test_total = 0
with torch.no_grad():
    for lip_frames, lbls in test_loader:
        lip_frames, lbls = lip_frames.to(device), lbls.to(device)
        outputs = model(lip_frames)
        _, predicted = torch.max(outputs, 1)
        test_total += lbls.size(0)
        test_correct += (predicted == lbls).sum().item()
test_accuracy = 100 * test_correct / test_total
print(f'Test Accuracy: {test_accuracy:.2f}%')