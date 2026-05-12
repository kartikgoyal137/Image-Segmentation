import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torchvision import transforms
from PIL import Image
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from models import AttentionUNet
from models import SwinUNet

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
TEST_DIR = os.path.join(BASE_DIR, "data/brisc/segmentation_task/test")
ATTENTION_WEIGHTS = os.path.join(BASE_DIR, "models/best_attention_unet.pth")
SWIN_WEIGHTS = os.path.join(BASE_DIR, "models/best_swin_unet.pth")
ENSEMBLE_SAVE_PATH = os.path.join(BASE_DIR, "models/best_weighted_ensemble.pth")
OUT_BASE = os.path.join(BASE_DIR, "data/brisc_processed/test_predictions")
OUT_MASKS = os.path.join(OUT_BASE, "masks")
OUT_OVERLAYS = os.path.join(OUT_BASE, "overlays")
os.makedirs(OUT_MASKS, exist_ok=True)
os.makedirs(OUT_OVERLAYS, exist_ok=True)
W_ATT = 0.45
W_SWIN = 0.55
IMG_SIZE = 256


class SimpleWeightedEnsemble(nn.Module):
    def __init__(self, att_model, swin_model, w_att=0.45, w_swin=0.55):
        super().__init__()
        self.att_model = att_model
        self.swin_model = swin_model
        self.register_buffer("w_att", torch.tensor(w_att))
        self.register_buffer("w_swin", torch.tensor(w_swin))

    def forward(self, img_att, img_swin):
        logits_att = self.att_model(img_att)
        logits_swin = self.swin_model(img_swin)
        prob_att = torch.sigmoid(logits_att)
        prob_swin = torch.sigmoid(logits_swin)
        return self.w_att * prob_att + self.w_swin * prob_swin


def preprocess_image_clahe(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read {img_path}")
    img_float = img.astype(np.float32)
    mean = np.mean(img_float)
    std = np.std(img_float)
    img_z = (img_float - mean) / (std + 1e-06)
    min_val = np.min(img_z)
    max_val = np.max(img_z)
    img_norm = (img_z - min_val) / (max_val - min_val + 1e-06)
    img_uint8 = (img_norm * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_uint8)
    return Image.fromarray(img_clahe)


class EnsembleTestDataset:
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
        img_pil = preprocess_image_clahe(img_path)
        mask = Image.open(msk_path).convert("L")
        img_att = self.att_tf(img_pil.copy())
        img_swin = self.swin_tf(img_pil.copy())
        mask_tensor = (self.msk_tf(mask) > 0.5).float()
        return img_att, img_swin, mask_tensor, img_pil, name


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


def create_overlay(img_pil, true_mask, pred_mask, target_size=(256, 256)):
    img_np = np.array(img_pil.resize(target_size)).astype(np.float32) / 255.0
    true_mask = true_mask[0].cpu().numpy()
    pred_mask = pred_mask[0].cpu().numpy()
    rgb = np.stack([img_np, img_np, img_np], axis=-1)
    rgb[true_mask == 1, 1] = np.clip(rgb[true_mask == 1, 1] + 0.5, 0, 1)
    rgb[pred_mask == 1, 0] = np.clip(rgb[pred_mask == 1, 0] + 0.5, 0, 1)
    return rgb


def save_binary_mask(pred_mask, out_path):
    mask_np = (pred_mask[0].cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(mask_np).save(out_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading Attention U-Net...")
    att_model = load_attention_model(device)
    print("Loading Swin U-Net...")
    swin_model = load_swin_model(device)
    ensemble_model = SimpleWeightedEnsemble(att_model, swin_model, W_ATT, W_SWIN).to(
        device
    )
    ensemble_model.eval()
    torch.save(
        {"model_state": ensemble_model.state_dict(), "w_att": W_ATT, "w_swin": W_SWIN},
        ENSEMBLE_SAVE_PATH,
    )
    print(f"Ensemble model saved to: {ENSEMBLE_SAVE_PATH}")
    dataset = EnsembleTestDataset(TEST_DIR, img_size=IMG_SIZE)
    print(f"Loaded {len(dataset)} test samples.")
    pbar = tqdm(dataset, desc="Evaluating Ensemble")
    for idx, (img_att, img_swin, true_mask, img_pil, name) in enumerate(pbar):
        img_att = img_att.unsqueeze(0).to(device)
        img_swin = img_swin.unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                prob_ensemble = ensemble_model(img_att, img_swin)
                pred_mask = (prob_ensemble > 0.5).float()[0]
        basename = os.path.splitext(name)[0]
        mask_out_path = os.path.join(OUT_MASKS, f"{basename}.png")
        overlay_out_path = os.path.join(OUT_OVERLAYS, f"{basename}_overlay.png")
        save_binary_mask(pred_mask, mask_out_path)
        overlay_img = create_overlay(img_pil, true_mask, pred_mask)
        plt.imsave(overlay_out_path, overlay_img)
    print(
        f"\nInference complete!\nMasks saved to: {OUT_MASKS}\nOverlays saved to: {OUT_OVERLAYS}"
    )


if __name__ == "__main__":
    main()
