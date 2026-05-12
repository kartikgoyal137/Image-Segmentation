import os
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, random_split
from models import (
    AttentionUNet,
    SwinUNet,
    FeatureLevelFusion,
    DisagreementWeightedLoss,
    MetaBRISCDataset,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_ROOT = os.path.join(BASE_DIR, "data")
ATTENTION_WEIGHTS = os.path.join(MODEL_DIR, "best_attention_unet.pth")
SWIN_WEIGHTS = os.path.join(MODEL_DIR, "best_swin_unet.pth")
SAVE_PATH = os.path.join(MODEL_DIR, "best_meta_fusion.pth")
NUM_EPOCHS = 20
BATCH_SIZE = 16
LR = 0.0001
IMG_SIZE = 256


def load_base_models(device):
    att_model = AttentionUNet(in_channels=1, base_ch=64).to(device)
    att_model.load_state_dict(
        torch.load(ATTENTION_WEIGHTS, map_location=device)["model_state"]
    )
    swin_model = SwinUNet(pretrained=False).to(device)
    swin_model.load_state_dict(
        torch.load(SWIN_WEIGHTS, map_location=device)["model_state"]
    )
    for m in [att_model, swin_model]:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
    return att_model, swin_model


def train_meta_learner():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
    att_model, swin_model = load_base_models(DEVICE)
    full_dataset = MetaBRISCDataset(
        os.path.join(DATA_ROOT, "brisc_processed/train"), img_size=IMG_SIZE
    )
    n_total = len(full_dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=16,
        persistent_workers=True,
        pin_memory=True,
    )
    meta_model = FeatureLevelFusion(att_ch=64, swin_ch=96, shared_dim=64).to(DEVICE)
    if hasattr(torch, "compile") and DEVICE.type == "cuda":
        try:
            meta_model = torch.compile(meta_model)
        except Exception:
            pass
    optimizer = torch.optim.AdamW(meta_model.parameters(), lr=LR, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = DisagreementWeightedLoss(k=2.0)
    scaler = torch.amp.GradScaler("cuda")
    best_dice = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        meta_model.train()
        epoch_loss = 0.0
        for img_att, img_swin, raw_img, masks in tqdm(
            train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"
        ):
            img_att, img_swin, raw_img, masks = [
                t.to(DEVICE, non_blocking=True)
                for t in [img_att, img_swin, raw_img, masks]
            ]
            optimizer.zero_grad()
            with torch.no_grad():
                out_att, feats_att = att_model(img_att, return_features=True)
                out_swin, feats_swin = swin_model(img_swin, return_features=True)
                probs_att = torch.sigmoid(out_att)
                probs_swin = torch.sigmoid(out_swin)
            with torch.amp.autocast("cuda"):
                logits = meta_model(feats_att, feats_swin, raw_img)
                loss = criterion(logits, masks, probs_att, probs_swin)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        scheduler.step()
        meta_model.eval()
        val_dice = 0.0
        with torch.inference_mode():
            for img_att, img_swin, raw_img, masks in val_loader:
                img_att, img_swin, raw_img, masks = [
                    t.to(DEVICE, non_blocking=True)
                    for t in [img_att, img_swin, raw_img, masks]
                ]
                _, feats_att = att_model(img_att, return_features=True)
                _, feats_swin = swin_model(img_swin, return_features=True)
                with torch.amp.autocast("cuda"):
                    logits = meta_model(feats_att, feats_swin, raw_img)
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    inter = (preds * masks).sum(dim=(2, 3))
                    union = preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
                    val_dice += (2 * inter / (union + 1e-06)).mean().item()
        avg_dice = val_dice / len(val_loader)
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Val Dice: {avg_dice:.4f}")
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(
                {"model_state": meta_model.state_dict(), "dice": avg_dice}, SAVE_PATH
            )
            print(f"Saved Best Model")


if __name__ == "__main__":
    train_meta_learner()
