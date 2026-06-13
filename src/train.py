import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import build_model, get_device
from src.preprocess import train_transforms, val_transforms

MODEL_PATH = "models/skin_detector.pth"
BATCH_SIZE = 16
EPOCHS_FASE1 = 10
EPOCHS_FASE2 = 10


class SkinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = int(row["label"])
        if self.transform:
            img = self.transform(img)
        return img, label


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def train():
    device = get_device()
    df = pd.read_csv("data/labels.csv")
    split = int(len(df) * 0.8)
    train_df, val_df = df[:split], df[split:]

    train_ds = SkinDataset(train_df, transform=train_transforms)
    val_ds   = SkinDataset(val_df,   transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    # Fase 1: solo la cabeza
    print("── Fase 1: Feature extraction ──")
    model = build_model(trainable_base=False).to(device)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    for epoch in range(EPOCHS_FASE1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc     = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{EPOCHS_FASE1} | loss: {train_loss:.3f} | acc: {train_acc:.3f} | val_acc: {val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  Modelo guardado (val_acc: {val_acc:.3f})")

    # Fase 2: fine-tuning completo
    print("\n── Fase 2: Fine-tuning ──")
    model = build_model(trainable_base=True).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    for epoch in range(EPOCHS_FASE2):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc     = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{EPOCHS_FASE2} | loss: {train_loss:.3f} | acc: {train_acc:.3f} | val_acc: {val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  Modelo guardado (val_acc: {val_acc:.3f})")

    print(f"\nEntrenamiento finalizado. Mejor accuracy: {best_acc:.3f}")


if __name__ == "__main__":
    train()