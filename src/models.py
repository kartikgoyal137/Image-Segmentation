import os
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False), nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False), nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.last_att = None

    def forward(self, g, x):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=True)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        att = self.psi(F.relu(g1 + x1, inplace=True))
        self.last_att = att
        return x * att


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.att = AttentionGate(F_g=in_ch // 2, F_l=skip_ch, F_int=skip_ch // 2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.att(g=x, x=skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, base_ch=64):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16)
        self.dec4 = UpBlock(base_ch * 16, base_ch * 8, base_ch * 8)
        self.dec3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.dec2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.dec1 = UpBlock(base_ch * 2, base_ch, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x, return_att=False, return_features=False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        out = self.out_conv(d1)
        if return_features:
            return out, d1
        if return_att:
            att_maps = [
                self.dec4.att.last_att,
                self.dec3.att.last_att,
                self.dec2.att.last_att,
                self.dec1.att.last_att,
            ]
            return out, att_maps
        return out


class SwinEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "swinv2_tiny_window8_256",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return [f.permute(0, 3, 1, 2).contiguous() for f in feats]


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class SwinUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = SwinEncoder(pretrained=pretrained)
        self.dec3 = DecoderBlock(in_ch=768, skip_ch=384, out_ch=384)
        self.dec2 = DecoderBlock(in_ch=384, skip_ch=192, out_ch=192)
        self.dec1 = DecoderBlock(in_ch=192, skip_ch=96, out_ch=96)
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def forward(self, x, return_features=False):
        e1, e2, e3, e4 = self.encoder(x)
        d3 = self.dec3(e4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        if return_features:
            return self.head(d1), d1
        return self.head(d1)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class FeatureLevelFusion(nn.Module):
    def __init__(self, att_ch=64, swin_ch=96, shared_dim=64):
        super().__init__()
        self.shared_dim = shared_dim
        self.proj_att = nn.Sequential(
            nn.Conv2d(att_ch, shared_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(shared_dim),
            nn.ReLU(inplace=True),
        )
        self.proj_swin = nn.Sequential(
            nn.Conv2d(swin_ch, shared_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(shared_dim),
            nn.ReLU(inplace=True),
        )
        self.se_att = SqueezeExcitation(shared_dim)
        self.se_swin = SqueezeExcitation(shared_dim)
        gate_in = shared_dim * 2 + 3
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(gate_in, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1),
            nn.Softmax(dim=1),
        )
        self.refine1 = ConvBlock(shared_dim, shared_dim)
        self.refine2 = ConvBlock(shared_dim, shared_dim)
        self.out_head = nn.Conv2d(shared_dim, 1, kernel_size=1)

    def forward(self, att_feats, swin_feats, raw_img):
        B, _, H, W = att_feats.shape
        device = att_feats.device
        swin_up = F.interpolate(
            swin_feats, size=(H, W), mode="bilinear", align_corners=True
        )
        p_att = self.proj_att(att_feats)
        p_swin = self.proj_swin(swin_up)
        p_att = self.se_att(p_att)
        p_swin = self.se_swin(p_swin)
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([gy, gx], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        raw_resized = F.interpolate(
            raw_img, size=(H, W), mode="bilinear", align_corners=True
        )
        gate_in = torch.cat([p_att, p_swin, coords, raw_resized], dim=1)
        weights = self.spatial_gate(gate_in)
        w_att = weights[:, 0:1, :, :]
        w_swin = weights[:, 1:2, :, :]
        fused = p_att * w_att + p_swin * w_swin
        residual = fused
        fused = self.refine1(fused)
        fused = self.refine2(fused)
        fused = fused + residual
        return self.out_head(fused)


class DisagreementWeightedLoss(nn.Module):
    def __init__(self, k=2.0):
        super().__init__()
        self.k = k
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets, prob_att, prob_swin):
        diff = torch.abs(prob_att - prob_swin)
        weight = 1.0 + self.k * diff
        bce_loss = (self.bce(logits, targets) * weight).mean()
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(2, 3))
        den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + 1e-06
        dice_loss = (1 - num / den).mean()
        return bce_loss + dice_loss


class MetaBRISCDataset(Dataset):
    def __init__(self, root, img_size=256):
        self.img_dir = os.path.join(root, "images")
        self.msk_dir = os.path.join(root, "masks")
        self.names = sorted(os.listdir(self.img_dir))
        self.img_size = img_size
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
        img_pil = Image.open(os.path.join(self.img_dir, name)).convert("L")
        mask_pil = Image.open(
            os.path.join(self.msk_dir, os.path.splitext(name)[0] + ".png")
        ).convert("L")
        return (
            self.att_tf(img_pil),
            self.swin_tf(img_pil),
            self.raw_tf(img_pil),
            (self.msk_tf(mask_pil) > 0.5).float(),
        )
