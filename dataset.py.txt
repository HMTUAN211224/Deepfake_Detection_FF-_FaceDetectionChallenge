from torch.utils.data import Dataset
import torch

class LipFrameDataset(Dataset):
    def __init__(self, lip_frames, labels):
        self.lip_frames = lip_frames
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        lip = self.lip_frames[idx]
        lip_tensor = torch.tensor(lip, dtype=torch.float32).permute(0, 3, 1, 2)
        if torch.rand(1) > 0.5:
            lip_tensor = torch.flip(lip_tensor, dims=[3])
        return lip_tensor, torch.tensor(self.labels[idx], dtype=torch.long)