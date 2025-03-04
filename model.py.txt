import torch
import torch.nn as nn

class LipDeepfakeDetector(nn.Module):
    def __init__(self, max_frames=10):
        super(LipDeepfakeDetector, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.gru = nn.GRU(512 * 4 * 4, 256, num_layers=2, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, 2)

    def forward(self, lip_frames):
        batch_size = lip_frames.size(0)
        cnn_out = self.cnn(lip_frames.contiguous().view(-1, 3, 64, 64)).view(batch_size, max_frames, -1)
        gru_out, _ = self.gru(cnn_out)
        gru_out = gru_out[:, -1, :]
        gru_out = self.dropout(gru_out)
        return self.fc(gru_out)