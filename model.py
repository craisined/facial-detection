import logging
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._inductor.config as inductor_config
import torchvision.transforms as T
from tqdm import tqdm
from sklearn.datasets import fetch_lfw_pairs
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision.datasets import ImageFolder

scaler = torch.amp.GradScaler("cuda")
logging.getLogger("torch._inductor").setLevel(logging.ERROR)


class LFWPairsDataset(Dataset):
    def __init__(self, size=224):
        data = fetch_lfw_pairs(
            subset="10_folds",
            funneled=True,
            color=True,
            resize=size / 130,
            slice_=(slice(65, 195), slice(60, 190)),
        )

        self.pairs = data.pairs
        self.targets = data.target

        self.pairs = np.transpose(self.pairs, (0, 1, 4, 2, 3))
        self.pairs = self.pairs.astype(np.float32)
        self.transform = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img1 = torch.tensor(self.pairs[idx, 0])
        img2 = torch.tensor(self.pairs[idx, 1])
        label = 1 if self.targets[idx] == 1 else -1
        return self.transform(img1), self.transform(img2), label


class PairedImageFolderDataset(Dataset):
    def __init__(self, size=112, root_dir="./data"):
        self.base_dataset = ImageFolder(root=root_dir)
        self.get_path = {}
        for idx, (path, label) in tqdm(
            enumerate(self.base_dataset.samples),
            desc="Building labels dict",
            unit="images",
            total=len(self.base_dataset),
        ):
            if label not in self.get_path:
                self.get_path[label] = []
            self.get_path[label].append(path)
        self.labels = list(self.get_path.keys())
        self.transform = T.Compose(
            [
                T.Resize((size, size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img1_path, img1_label = self.base_dataset.samples[idx]
        img1 = self.base_dataset.loader(img1_path)
        similar = random.choice((True, False))
        img2_label = img1_label
        while not similar and img2_label == img1_label:
            img2_label = random.choice(self.labels)
        img2_path = random.choice(self.get_path[img2_label])
        img2 = self.base_dataset.loader(img2_path)
        return self.transform(img1), self.transform(img2), (1 if similar else -1)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.block(x)
        shortcut = self.shortcut(x)
        return self.relu(residual + shortcut)


class FacialRecognitionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64, 64),
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
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128, bias=False),
        )

    def forward(self, face_tensor):
        unnormalized = self.resnet(face_tensor)
        return F.normalize(unnormalized, p=2, dim=1)


def load_data(batch_size=256, size=112):
    dataset = PairedImageFolderDataset(size=size)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataloader


def load_lfw_data(batch_size=256, size=112):
    dataset = LFWPairsDataset(size=size)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataloader


def train_epoch(model, data, optimizer, device="cuda"):
    model.train()
    loss_func = nn.CosineEmbeddingLoss(margin=0.3)
    accum_loss = 0
    pbar = tqdm(enumerate(data), unit="batches", total=len(data), desc="Training: ")
    for batch_idx, (face_1, face_2, target) in pbar:
        face_1 = face_1.to(device, non_blocking=True, memory_format=torch.channels_last)
        face_2 = face_2.to(device, non_blocking=True, memory_format=torch.channels_last)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            emb1, emb2 = model(torch.cat([face_1, face_2])).chunk(2)
        loss = loss_func(emb1, emb2, target)
        accum_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        pbar.set_postfix(loss=f"{accum_loss / (batch_idx + 1):.4f}")
    return accum_loss / len(data)


def validate(model, data, device="cuda"):
    model.eval()
    accum_loss = 0
    loss_func = nn.CosineEmbeddingLoss(margin=0.3)
    all_sims, all_targets = [], []
    with torch.no_grad():
        pbar = tqdm(
            enumerate(data), unit="batches", total=len(data), desc="Validating: "
        )
        for batch_idx, (face_1, face_2, target) in pbar:
            face_1 = face_1.to(
                device, non_blocking=True, memory_format=torch.channels_last
            )
            face_2 = face_2.to(
                device, non_blocking=True, memory_format=torch.channels_last
            )
            target = target.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                emb1, emb2 = model(torch.cat([face_1, face_2])).chunk(2)
            loss = loss_func(emb1, emb2, target)
            similarity = torch.sum(emb1 * emb2, dim=1)
            accum_loss += loss.item()
            all_sims.append(torch.sum(emb1 * emb2, dim=1).cpu())
            all_targets.append(target.cpu())
            pbar.set_postfix(loss=f"{accum_loss / (batch_idx + 1):.4f}")

    sims = torch.cat(all_sims)
    targets = torch.cat(all_targets)
    thresholds = torch.linspace(-1, 1, steps=100)
    best_acc = 0
    best_threshold = 0
    for threshold in thresholds:
        predictions = torch.where(sims > threshold, 1.0, -1.0)
        acc = torch.sum(predictions == targets).item() / len(targets)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold.item()

    return (
        accum_loss / len(data),
        best_acc,
        best_threshold,
    )


if __name__ == "__main__":
    checkpoint = "checkpoint.pth"
    start_epochs = 0
    total_epochs = 20
    model = FacialRecognitionModel()
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model = model.to("cuda", memory_format=torch.channels_last)
    model = torch.compile(model)
    no_decay_params = [
        p for p in model.parameters() if p.ndim == 1
    ]  # For avoiding weight decay on bias and batch norm parameters
    decay_params = [p for p in model.parameters() if p.ndim > 1]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 1e-2},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_epochs, 
        eta_min=1e-6
    )
    for _ in range(start_epochs):
        scheduler.step()
    train_set, test_set = load_data(), load_lfw_data()
    print("Data loaded!")
    val_loss, val_acc, threshold = validate(model, test_set)
    print(
        f"Epoch 0: {val_loss:.4f} (val) | 0.0000 (train) | {val_acc:.4f} (val acc) | {threshold:.4f} (threshold)"
    )
    for epoch in range(start_epochs, total_epochs):
        train_loss = train_epoch(model, train_set, optimizer)
        scheduler.step()
        val_loss, val_acc, threshold = validate(model, test_set)
        print(
            f"Epoch {epoch + 1}: {val_loss:.4f} (val) | {train_loss:.4f} (train) | {val_acc:.4f} (val acc) | {threshold:.4f} (threshold)"
        )
        torch.save(model._orig_mod.state_dict(), f"epoch_{epoch + 1}.pth")
