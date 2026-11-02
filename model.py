import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import LFWPairs
from torch.utils.data import DataLoader


class FacialRecognitionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.linear_head = LinearHead()

    def forward(self, face_1, face_2):
        embedding_1 = self.encoder(face_1)
        embedding_2 = self.encoder(face_2)
        return self.linear_head(embedding_1, embedding_2)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.block(x)
        shortcut = self.shortcut(x)
        return self.relu(residual + shortcut)


class Encoder(nn.Module):
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
        )

    def forward(self, face_tensor):
        return self.resnet(face_tensor)


class LinearHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, face_encoding_1, face_encoding_2):
        return self.mlp(torch.cat((face_encoding_1, face_encoding_2), dim=1))


def load_data(split, batch_size=64):
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
        ]
    )
    dataset = LFWPairs(
        root="./data",
        split=split,
        image_set="deepfunneled",
        transform=transform,
        download=False,
    )
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
    loss_func = nn.BCEWithLogitsLoss()
    accum_loss = 0
    for batch_idx, (face_1, face_2, target) in enumerate(data):
        face_1 = face_1.to(device)
        face_2 = face_2.to(device)
        target = target.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        logits = model(face_1, face_2)
        loss = loss_func(logits, target)
        accum_loss += loss.item()
        loss.backward()
        optimizer.step()
    return accum_loss / len(data)


def validate(model, data, device="cuda"):
    model.eval()
    accum_loss = 0
    loss_func = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch_idx, (face_1, face_2, target) in enumerate(data):
            face_1 = face_1.to(device)
            face_2 = face_2.to(device)
            target = target.to(device).float().unsqueeze(1)
            logits = model(face_1, face_2)
            accum_loss += loss_func(logits, target).item()
    return accum_loss / len(data)


if __name__ == "__main__":
    epochs = 10
    model = FacialRecognitionModel()
    model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_set, test_set = load_data("train"), load_data("test")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_set, optimizer)
        val_loss = validate(model, test_set)
        print(f"Epoch {epoch + 1}: {val_loss} (val) | {train_loss} (train)")
        torch.save(model.state_dict(), f'epoch_{epoch + 1}.pth')