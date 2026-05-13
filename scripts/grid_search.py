import os
import torch
import numpy as np
from tqdm.auto import tqdm
from torchvision import transforms
from PIL import Image
import sys

sys.path.append("../src")
from models import AttentionUNet
from models import SwinUNet

VAL_DIR = "./brain_brisc/data/brisc_processed/val"
ATTENTION_WEIGHTS = (
    "./brain_brisc/models/best_attention_unet.pth"
)
SWIN_WEIGHTS = "./brain_brisc/models/best_swin_unet.pth"
IMG_SIZE = 256


class EnsembleValDataset:
    def __init__(self, root, img_size=256):
        self.img_dir = os.path.join(root, "images")
        self.msk_dir = os.path.join(root, "masks")
        self.names = sorted(os.listdir(self.img_dir))
        self.att_tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        self.swin_tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.msk_tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = os.path.join(self.img_dir, name)
        msk_path = os.path.join(self.msk_dir, os.path.splitext(name)[0] + ".png")
        img_pil = Image.open(img_path).convert("L")
        mask = Image.open(msk_path).convert("L")
        img_att = self.att_tf(img_pil.copy())
        img_swin = self.swin_tf(img_pil.copy())
        mask_tensor = self.msk_tf(mask)
        mask_tensor = (mask_tensor > 0.5).float()
        return img_att, img_swin, mask_tensor


def load_attention_model(device):
    model = AttentionUNet(in_channels=1, base_ch=64).to(device)
    checkpoint = torch.load(ATTENTION_WEIGHTS, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_swin_model(device):
    model = SwinUNet(pretrained=False).to(device)
    checkpoint = torch.load(SWIN_WEIGHTS, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading Attention U-Net...")
    att_model = load_attention_model(device)
    print("Loading Swin U-Net...")
    swin_model = load_swin_model(device)
    dataset = EnsembleValDataset(VAL_DIR, img_size=IMG_SIZE)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=4
    )
    print(f"Loaded {len(dataset)} validation samples.")
    all_prob_att = []
    all_prob_swin = []
    all_masks = []
    with torch.no_grad():
        for img_att, img_swin, true_mask in tqdm(
            loader, desc="Predicting validation set"
        ):
            img_att = img_att.to(device)
            img_swin = img_swin.to(device)
            with torch.cuda.amp.autocast():
                logits_att = att_model(img_att)
                logits_swin = swin_model(img_swin)
                prob_att = torch.sigmoid(logits_att).cpu()
                prob_swin = torch.sigmoid(logits_swin).cpu()
            all_prob_att.append(prob_att)
            all_prob_swin.append(prob_swin)
            all_masks.append(true_mask)
    all_prob_att = torch.cat(all_prob_att, dim=0)
    all_prob_swin = torch.cat(all_prob_swin, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    w_att_range = np.linspace(0.0, 1.0, 21)
    best_w_att = 0.0
    best_w_swin = 1.0
    best_dice = 0.0
    print("\nGrid Search over w_att and w_swin...")
    for w_att in w_att_range:
        w_swin = 1.0 - w_att
        prob_ensemble = w_att * all_prob_att + w_swin * all_prob_swin
        preds = (prob_ensemble > 0.5).float()
        num = 2 * (preds * all_masks).sum(dim=(1, 2, 3))
        den = preds.sum(dim=(1, 2, 3)) + all_masks.sum(dim=(1, 2, 3)) + 1e-06
        dice = (num / den).mean().item()
        print(f"w_att: {w_att:.2f}, w_swin: {w_swin:.2f} => Dice: {dice:.4f}")
        if dice > best_dice:
            best_dice = dice
            best_w_att = w_att
            best_w_swin = w_swin
    print(
        f"""
Optimal Weights: w_att = {best_w_att:.2f}, w_swin = {best_w_swin:.2f} (Val Dice: {best_dice:.4f})"""
    )


if __name__ == "__main__":
    main()
