import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_lfw_pairs
from torch.utils.data import Dataset, DataLoader


class LFWPairsDataset(Dataset):
    def __init__(self, subset, size=224):
        data = fetch_lfw_pairs(subset=subset, color=True, resize=size / 250)

        self.pairs = data.pairs
        self.targets = data.target

        self.pairs = np.transpose(self.pairs, (0, 1, 4, 2, 3))
        self.pairs = self.pairs.astype(np.float32) / 255.0

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img1 = torch.tensor(self.pairs[idx, 0])
        img2 = torch.tensor(self.pairs[idx, 1])
        label = torch.tensor(1 if self.targets[idx] == 1 else -1, dtype=torch.float32)
        return img1, img2, label


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.block(x)
        shortcut = self.shortcut(x)
        return self.relu(residual + shortcut)


class FacialRecognitionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = nn.Sequential(
            ResBlock(3, 64),
            ResBlock(64, 64),
            nn.MaxPool2d(2),
            ResBlock(64, 128),
            ResBlock(128, 128),
            nn.MaxPool2d(2),
            ResBlock(128, 256),
            ResBlock(256, 256),
            nn.MaxPool2d(2),
            ResBlock(256, 512),
            ResBlock(512, 512),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, face_tensor):
        return self.resnet(face_tensor)


# TODO: switch to scikit-learn's lfw
def load_data(split, batch_size=32):
    dataset = LFWPairsDataset(subset=split)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=True,
    )
    return dataloader


def train_epoch(model, data, optimizer, device="cuda"):
    model.train()
    loss_func = nn.CosineEmbeddingLoss()
    accum_loss = 0
    for batch_idx, (face_1, face_2, target) in enumerate(data):
        face_1, face_2, target = face_1.to(device), face_2.to(device), target.to(device)
        optimizer.zero_grad()
        embeddings = model(face_1), model(face_2)
        loss = loss_func(embeddings[0], embeddings[1], target)
        accum_loss += loss.item()
        loss.backward()
        optimizer.step()
    return accum_loss / len(data)


def validate(model, data, device="cuda"):
    model.eval()
    accum_loss = 0
    loss_func = nn.CosineEmbeddingLoss()
    with torch.no_grad():
        for batch_idx, (face_1, face_2, target) in enumerate(data):
            face_1, face_2, target = face_1.to(device), face_2.to(device), target.to(device)
            embeddings = model(face_1), model(face_2)
            loss = loss_func(embeddings[0], embeddings[1], target)
            accum_loss += loss.item()
    return accum_loss / len(data)


if __name__ == "__main__":
    epochs = 10
    model = FacialRecognitionModel()
    model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    train_set, test_set = load_data("train"), load_data("test")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_set, optimizer)
        val_loss = validate(model, test_set)
        print(f"Epoch {epoch + 1}: {val_loss} (val) | {train_loss} (train)")
        torch.save(model.state_dict(), f"epoch_{epoch + 1}.pth")
