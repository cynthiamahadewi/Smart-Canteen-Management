import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DATA_ROOT = os.path.join(os.path.dirname(__file__),
                         "LeFood-Set Leftovers Food Dataset/LeFood-Set/leftover dataset")
EXCEL_PATH = os.path.join(os.path.dirname(__file__),
                          "LeFood-Set Leftovers Food Dataset/LeFood-Set/data_original.xlsx")

# 5 ordinal bins: [0,0.2), [0.2,0.4), [0.4,0.6), [0.6,0.8), [0.8,1.0]
BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
NUM_CLASSES = 5


def ratio_to_ordinal(r: float) -> int:
    for i in range(len(BINS) - 1):
        if BINS[i] <= r < BINS[i + 1]:
            return i
    return NUM_CLASSES - 1


class LeFoodDataset(Dataset):
    def __init__(self, split="train", transform=None, val_ratio=0.15, test_ratio=0.15, seed=42):
        df = pd.read_excel(EXCEL_PATH)

        # compute consumption ratio from weights
        df = df.dropna(subset=["Weight Before Eaten (g)", "Weight After Eaten (g)"])
        df = df[df["Weight Before Eaten (g)"] > 0].copy()
        df["r"] = 1.0 - df["Weight After Eaten (g)"] / df["Weight Before Eaten (g)"]
        df["r"] = df["r"].clip(0.0, 1.0)
        df["ordinal"] = df["r"].apply(ratio_to_ordinal)

        # resolve image paths: first 3 chars of filename = subfolder
        def resolve_path(filename, split_dir):
            folder = str(filename)[:3]
            return os.path.join(DATA_ROOT, split_dir, folder, filename)

        df["after_path"] = df["Image After Eaten"].apply(
            lambda f: resolve_path(f, "data_after"))
        df["before_path"] = df["Image Before Eaten"].apply(
            lambda f: resolve_path(f, "data_before"))

        # drop rows where either image file doesn't exist
        df = df[df["after_path"].apply(os.path.exists) &
                df["before_path"].apply(os.path.exists)].reset_index(drop=True)

        # train/val/test split (stratified by food category = folder)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(df))
        n = len(df)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)

        if split == "test":
            df = df.iloc[idx[:n_test]]
        elif split == "val":
            df = df.iloc[idx[n_test:n_test + n_val]]
        else:
            df = df.iloc[idx[n_test + n_val:]]

        self.df = df.reset_index(drop=True)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        # before image: deterministic transform only — it serves as a fixed reference,
        # spatial augmentation on the reference would break the before/after visual comparison.
        self.before_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["after_path"]).convert("RGB")
        img = self.transform(img)
        img_before = Image.open(row["before_path"]).convert("RGB")
        img_before = self.before_transform(img_before)
        return {
            "image": img,
            "before": img_before,
            "r": float(row["r"]),
            "ordinal": int(row["ordinal"]),
            "food_name": row["Name of the food"],
        }


def get_dataloaders(batch_size=16, num_workers=0):
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = LeFoodDataset(split="train", transform=train_transform)
    val_ds = LeFoodDataset(split="val", transform=val_transform)
    test_ds = LeFoodDataset(split="test", transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    print(f"Dataset split — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
