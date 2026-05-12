import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torchvision import transforms
from PIL import Image
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from models import AttentionUNet, SwinUNet, FeatureLevelFusion

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
TEST_DIR = os.path.join(BASE_DIR, "data/brisc/segmentation_task/test")
ATTENTION_WEIGHTS = os.path.join(BASE_DIR, "models/best_attention_unet.pth")
SWIN_WEIGHTS = os.path.join(BASE_DIR, "models/best_swin_unet.pth")
META_WEIGHTS = os.path.join(BASE_DIR, "models/best_meta_fusion.pth")
OUT_BASE = os.path.join(BASE_DIR, "data/brisc_processed/fusion_predictions")
OUT_MASKS = os.path.join(OUT_BASE, "masks")
OUT_OVERLAYS = os.path.join(OUT_BASE, "overlays")
os.makedirs(OUT_MASKS, exist_ok=True)
os.makedirs(OUT_OVERLAYS, exist_ok=True)
IMG_SIZE = 256


def preprocess_image_clahe(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read {img_path}")
    img_float = img.astype(np.float32)
    mean = np.mean(img_float)
    std = np.std(img_float)
    if std > 0:
        img_z = (img_float - mean) / std
    else:
        img_z = img_float - mean
    min_val = np.min(img_z)
    max_val = np.max(img_z)
    if max_val - min_val > 0:
        img_norm = (img_z - min_val) / (max_val - min_val)
    else:
        img_norm = img_z
    img_uint8 = (img_norm * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_uint8)
    return Image.fromarray(img_clahe)


class FusionTestDataset:
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
        self.raw_tf = transforms.Compose(
            [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
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
        raw_img = self.raw_tf(img_pil.copy())
        mask_tensor = (self.msk_tf(mask) > 0.5).float()
        return img_att, img_swin, raw_img, mask_tensor, img_pil, name


def load_attention_model(device):
    model = AttentionUNet(in_channels=1, base_ch=64).to(device)
    if not os.path.exists(ATTENTION_WEIGHTS):
        raise FileNotFoundError(f"Missing {ATTENTION_WEIGHTS}")
    checkpoint = torch.load(ATTENTION_WEIGHTS, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_swin_model(device):
    model = SwinUNet(pretrained=False).to(device)
    if not os.path.exists(SWIN_WEIGHTS):
        raise FileNotFoundError(f"Missing {SWIN_WEIGHTS}")
    checkpoint = torch.load(SWIN_WEIGHTS, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_meta_model(device):
    model = FeatureLevelFusion(att_ch=64, swin_ch=96, shared_dim=64).to(device)
    if not os.path.exists(META_WEIGHTS):
        raise FileNotFoundError(f"Missing {META_WEIGHTS}")
    checkpoint = torch.load(META_WEIGHTS, map_location=device)
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
    print("Loading base models and meta-learner...")
    att_model = load_attention_model(device)
    swin_model = load_swin_model(device)
    meta_model = load_meta_model(device)
    dataset = FusionTestDataset(TEST_DIR, img_size=IMG_SIZE)
    print(f"Loaded {len(dataset)} test samples.")
    pbar = tqdm(dataset, desc="Fusion Inference")
    for idx, (img_att, img_swin, raw_img, true_mask, img_pil, name) in enumerate(pbar):
        img_att_batch = img_att.unsqueeze(0).to(device)
        img_swin_batch = img_swin.unsqueeze(0).to(device)
        raw_img_batch = raw_img.unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
                _, feats_att = att_model(img_att_batch, return_features=True)
                _, feats_swin = swin_model(img_swin_batch, return_features=True)
                logits = meta_model(feats_att, feats_swin, raw_img_batch)
                probs = torch.sigmoid(logits)
                pred_mask = (probs > 0.5).float()[0]
        basename = os.path.splitext(name)[0]
        mask_out_path = os.path.join(OUT_MASKS, f"{basename}.png")
        overlay_out_path = os.path.join(OUT_OVERLAYS, f"{basename}_overlay.png")
        save_binary_mask(pred_mask, mask_out_path)
        overlay_img = create_overlay(img_pil, true_mask, pred_mask)
        plt.imsave(overlay_out_path, overlay_img)
    print(f"\nFusion Inference complete!")
    print(f"Binary masks saved to: {OUT_MASKS}")
    print(f"Overlaid images saved to: {OUT_OVERLAYS}")


if __name__ == "__main__":
    main()
