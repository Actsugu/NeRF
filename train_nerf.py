import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


#设置与超参数
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'data' / 'nerf_synthetic' / 'lego'
OUTPUT_PATH = SCRIPT_DIR / 'runs' / 'nerf_lego.pt'

TRAIN_STEPS = 200_000
RAYS_PER_STEP = 1024
LEARNING_RATE = 5e-4
LR_DECAY_STEPS = 250_000
LR_DECAY_FACTOR = 0.1
LOG_EVERY = 100
SEED = 0
DEVICE = "cuda"

IMAGE_DOWNSCALE = 2
NEAR, FAR = 2.0, 6.0
N_COARSE, N_FINE = 64, 64
POSITION_BANDS, DIRECTION_BANDS = 10, 4
MLP_DEPTH, MLP_WIDTH, VIEW_WIDTH = 8, 256, 128
SKIP_LAYER = 5  # 从 0 起算，默认第 6 层接收跳连


#从本地读取数据
def load_blender_data(data_dir):

    metadata_path = data_dir / "transforms_train.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing {metadata_path}. Extract the official Lego data and set DATA_DIR."
        )
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    frames = metadata["frames"]
    if not frames:
        raise ValueError(f"No training frames found in {metadata_path}")

    images = []
    scale = IMAGE_DOWNSCALE
    if scale < 1:
        raise ValueError("IMAGE_DOWNSCALE must be positive")
    for frame in frames:
        path = (data_dir / frame["file_path"]).with_suffix(".png")
        if not path.is_file():
            raise FileNotFoundError(f"Missing training image {path}. Check the extracted Lego data.")
        with Image.open(path) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0

        # 按面积平均降采样，再合成白色背景。
        height, width = rgba.shape[:2]
        if height % scale or width % scale:
            raise ValueError(f"IMAGE_DOWNSCALE must divide both image dimensions ({width}x{height})")
        if scale > 1:
            rgba = rgba.reshape(height // scale, scale, width // scale, scale, 4).mean(axis=(1, 3))
        alpha = rgba[..., 3:4]
        images.append(rgba[..., :3] * alpha + (1.0 - alpha))

    images = torch.from_numpy(np.stack(images))
    poses = torch.tensor([frame["transform_matrix"] for frame in frames], dtype=torch.float32)
    width = images.shape[2]
    focal = 0.5 * width / math.tan(0.5 * metadata["camera_angle_x"])
    return images, poses, focal


def rays_from_pixels(x, y, height, width, focal, pose):
    # Blender/OpenGL 相机看向负 Z 轴，图像行号向下、相机 Y 轴向上。
    directions = torch.stack(
        ((x - 0.5 * width) / focal, (0.5 * height - y) / focal, -torch.ones_like(x)),
        dim=-1,
    )
    return pose[:3, 3], directions @ pose[:3, :3].T

#模型架构
class PositionalEncoding(nn.Module):
    def __init__(self, bands):
        super().__init__()
        self.out_dim = 3 * (1 + 2 * bands)
        # 跟随 TensorFlow 实现，保留原始坐标，频率为 2^k。
        self.register_buffer("frequencies", 2.0 ** torch.arange(bands))

    def forward(self, x):
        angles = x.unsqueeze(-2) * self.frequencies[:, None]
        periodic = torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        return torch.cat((x, periodic), dim=-1)


class NeRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.position = PositionalEncoding(POSITION_BANDS)
        self.direction = PositionalEncoding(DIRECTION_BANDS)
        position_dim = self.position.out_dim
        self.layers = nn.ModuleList([nn.Linear(position_dim, MLP_WIDTH)])
        for index in range(1, MLP_DEPTH):
            in_dim = MLP_WIDTH + position_dim if index == SKIP_LAYER else MLP_WIDTH
            self.layers.append(nn.Linear(in_dim, MLP_WIDTH))

        self.density = nn.Linear(MLP_WIDTH, 1)
        self.features = nn.Linear(MLP_WIDTH, MLP_WIDTH)
        self.view_layer = nn.Linear(MLP_WIDTH + self.direction.out_dim, VIEW_WIDTH)
        self.rgb = nn.Linear(VIEW_WIDTH, 3)

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, points, view_dirs):

        position = self.position(points)
        hidden = position
        for index, layer in enumerate(self.layers):
            if index == SKIP_LAYER:
                hidden = torch.cat((position, hidden), dim=-1)
            hidden = F.relu(layer(hidden))

        density = self.density(hidden).squeeze(-1)
        direction = self.direction(view_dirs)
        direction = direction[:, None, :].expand(-1, points.shape[1], -1)
        color_input = torch.cat((self.features(hidden), direction), dim=-1)
        rgb = self.rgb(F.relu(self.view_layer(color_input)))
        return rgb, density


def render_samples(model, ray_origin, ray_dirs, ray_lengths, view_dirs, depths):
    points = ray_origin + ray_dirs[:, None, :] * depths[..., None]
    rgb, density = model(points, view_dirs)

    intervals = depths[:, 1:] - depths[:, :-1]
    intervals = torch.cat((intervals, torch.full_like(depths[:, :1], 1e10)), dim=-1)
    distances = intervals * ray_lengths
    optical_depth = F.relu(density) * distances

    #透射率只累计到当前采样点之前，剩余透射率对应白色背景
    alpha = -torch.expm1(-optical_depth)
    accumulated = torch.cumsum(optical_depth[:, :-1], dim=-1)
    transmittance = torch.exp(-F.pad(accumulated, (1, 0)))
    weights = alpha * transmittance
    color = (weights[..., None] * rgb.sigmoid()).sum(dim=1)
    color = color + (1.0 - weights.sum(dim=1, keepdim=True))
    return color, weights


@torch.no_grad()
def sample_pdf(bins, weights, count):
    #反演粗采样权重的CDF，为细网络选择额外的深度
    weights = weights + 1e-5
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = F.pad(pdf.cumsum(dim=-1), (1, 0))
    u = torch.rand((bins.shape[0], count), device=bins.device, dtype=bins.dtype)

    upper = torch.searchsorted(cdf, u, right=True)
    lower = (upper - 1).clamp_min(0)
    upper = upper.clamp_max(cdf.shape[-1] - 1)
    cdf_low, cdf_high = cdf.gather(1, lower), cdf.gather(1, upper)
    bin_low, bin_high = bins.gather(1, lower), bins.gather(1, upper)
    fraction = (u - cdf_low) / (cdf_high - cdf_low).clamp_min(1e-8)
    return torch.lerp(bin_low, bin_high, fraction)


def render_rays(coarse, fine, ray_origin, ray_dirs):
    ray_lengths = torch.linalg.vector_norm(ray_dirs, dim=-1, keepdim=True)
    view_dirs = ray_dirs / ray_lengths

    # 先在各深度区间内随机采一点，再按粗网络的体渲染权重细采样。
    depths = torch.linspace(NEAR, FAR, N_COARSE, device=ray_dirs.device, dtype=ray_dirs.dtype)
    midpoints = 0.5 * (depths[:-1] + depths[1:])
    lower = torch.cat((depths[:1], midpoints))
    upper = torch.cat((midpoints, depths[-1:]))
    jitter = torch.rand((ray_dirs.shape[0], N_COARSE), device=ray_dirs.device, dtype=ray_dirs.dtype)
    depths = torch.lerp(lower, upper, jitter)
    coarse_rgb, weights = render_samples(coarse, ray_origin, ray_dirs, ray_lengths, view_dirs, depths)

    bins = 0.5 * (depths[:, :-1] + depths[:, 1:])
    extra = sample_pdf(bins, weights[:, 1:-1], N_FINE)
    fine_depths = torch.cat((depths, extra), dim=-1).sort(dim=-1).values
    fine_rgb, _ = render_samples(fine, ray_origin, ray_dirs, ray_lengths, view_dirs, fine_depths)
    return coarse_rgb, fine_rgb

#训练
def run(data_dir, steps, rays, output):
    if steps < 1 or rays < 1 or LOG_EVERY < 1:
        raise ValueError("TRAIN_STEPS, RAYS_PER_STEP and LOG_EVERY must be positive")
    if N_COARSE < 3 or N_FINE < 1 or not 0 < NEAR < FAR:
        raise ValueError("Set N_COARSE >= 3, N_FINE >= 1 and 0 < NEAR < FAR")
    if not 1 <= SKIP_LAYER < MLP_DEPTH:
        raise ValueError("SKIP_LAYER must be between 1 and MLP_DEPTH - 1")


    data_dir, output = Path(data_dir), Path(output)

    random.seed(SEED)
    torch.manual_seed(SEED)
    device_name = DEVICE
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    images, poses, focal = load_blender_data(data_dir)
    num_images, height, width, _ = images.shape
    if rays > height * width:
        raise ValueError(f"RAYS_PER_STEP must be at most {height * width}")
    images, poses = images.to(device), poses.to(device)
    print(f"{data_dir}: {num_images} train images, {width}x{height}, device={device}", flush=True)

    coarse, fine = NeRF().to(device), NeRF().to(device)
    optimizer = torch.optim.Adam([*coarse.parameters(), *fine.parameters()], lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=LR_DECAY_FACTOR ** (1 / LR_DECAY_STEPS)
    )

    start = time.perf_counter()
    for step in range(1, steps + 1):
        view = random.randrange(num_images)
        pixels = torch.randperm(height * width, device=device)[:rays]
        y, x = pixels // width, pixels % width
        target_rgb = images[view, y, x]
        origin, directions = rays_from_pixels(x, y, height, width, focal, poses[view])

        optimizer.zero_grad(set_to_none=True)
        coarse_rgb, fine_rgb = render_rays(coarse, fine, origin, directions)
        fine_mse = F.mse_loss(fine_rgb, target_rgb)
        loss = F.mse_loss(coarse_rgb, target_rgb) + fine_mse
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step == 1 or step % LOG_EVERY == 0 or step == steps:
            psnr = -10.0 * math.log10(max(fine_mse.item(), 1e-10))
            elapsed = time.perf_counter() - start
            print(f"step {step:6d}/{steps}  loss={loss.item():.6f}  PSNR={psnr:.2f} dB  {elapsed:.1f}s", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"coarse": coarse.state_dict(), "fine": fine.state_dict(), "steps": steps}, output)
    print(f"Saved weights to {output}", flush=True)


if __name__ == "__main__":
    run(DATA_DIR, TRAIN_STEPS, RAYS_PER_STEP, OUTPUT_PATH)