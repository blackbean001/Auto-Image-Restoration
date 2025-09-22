from pathlib import Path
import numpy as np
import cv2
from scipy.io import loadmat
import torch
import math
import shutil
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.utils.matlab_functions import imresize


__all__ = [
    "lr", 
    "darken", 
    "add_noise", 
    "add_jpeg_comp_artifacts", 
    "add_haze",
    "add_motion_blur",
    "add_defocus_blur",
    "add_rain",
]


def lr(img, level=4):
    """Resizes the image to 1/4 of its original size."""
    img = img.copy()

    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = imresize(img, scale=0.25)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).round().astype(np.uint8)

    return img


def add_noise(img, level: int, noise_type: str | None = None):
    """Adds Gaussian or Poisson noise to the image."""

    img = img.copy()
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    types = ["Gaussian", "Poisson"]
    if noise_type is None:
        noise_type = np.random.choice(types)
    else:
        assert noise_type in types

    if noise_type == "Gaussian":
        sigma_range = [
            [2, 6], [10, 20], [25, 35], [40, 50]
        ][level-1]
        out = random_add_gaussian_noise_pt(
            img, 
            sigma_range=sigma_range, 
            # gray_prob=gray_noise_prob,
            clip=True, 
            rounds=False)
        
    else:
        scale_range = [
            [0.4, 0.6], [0.8, 1.2], [1.7, 2.3], [2.7, 3.3]
        ][level-1]
        out = random_add_poisson_noise_pt(
            img,
            scale_range=scale_range,
            # gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)
        
    lq = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    lq = (lq * 255).clip(0,255).round().astype(np.uint8)
    return lq


def add_jpeg_comp_artifacts(img, level: int):
    """Applies JPEG compression. `quality_factor: int` in [10, 30)."""    

    img = img.copy()
    mean_qf = [50, 30, 20, 10]
    quality_factor = np.random.normal(mean_qf[level-1], 2)
    quality_factor = max(min(round(quality_factor), 100), 5)
    _, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    return img


def darken(img, level: int, darken_type: str | None = None):
    """Darkens the image by one of three methods: constant shift, gamma correction, linear mapping."""    

    img = img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    types = ["constant shift", "gamma correction", "linear mapping"]
    if darken_type is None:
        darken_type = np.random.choice(types)
    else:
        assert darken_type in types

    if darken_type == "constant shift":
        # shift in [30, 50)
        mean_shift = [15, 30, 50, 70]
        shift = np.random.normal(mean_shift[level-1], 2)
        shift = max(round(shift), 5)
        v = np.clip(np.int16(v)-shift, 0, 255).round().astype(np.uint8)
    elif darken_type == "gamma correction":
        # gamma in [0.5, 0.7)
        mean_gamma = [0.8, 0.7, 0.6, 0.5]
        gamma = np.random.normal(mean_gamma[level-1], 0.015)
        gamma = min(gamma, 0.95)
        v = (cv2.pow(v / 255.0, 1.0 / gamma) * 255).clip(0,255).round().astype(np.uint8)
    else:
        # compress v to 0-dst_max in [100, 150)
        vmin, vmax = np.min(v), np.max(v)
        mean_dst_max = [230, 190, 150, 120]
        dst_max = np.random.normal(mean_dst_max[level-1], 4)
        dst_max = max(min(round(dst_max), vmax-5), 80)
        v = ((v - vmin) / (vmax - vmin) * dst_max).round().astype(np.uint8)
    
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def add_haze(img, idx, level, depth_dir=Path("training_data/depth")):
    """Adds haze by atmospheric scattering model I(x) = J(x)t(x) + A(1-t(x)), where t(x) = exp(-beta d(x)). This code is adapted from that of [MiOIR](https://github.com/Xiangtaokong/MiOIR). Following [RESIDE](https://ieeexplore.ieee.org/document/8451944), A ~ U(0.7, 1.0), beta ~ U(0.6, 1.8)."""

    img = img.copy()

    d = loadmat(depth_dir/idx/"predict_depth.mat")
    d = d['data_obj']
    d = cv2.resize(d, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    d = d / d.max()

    A_mean, beta_mean = [
        [0.6, 0.4], [0.7, 0.6], [0.8, 1.2], [1.0, 1.8]
    ][level-1]
    A = max(np.random.normal(A_mean, 0.015), 0.4)
    beta = np.random.normal(beta_mean, 0.05)

    t = np.exp(-beta * d)
    t = t[..., np.newaxis]

    return (img * t + A * 255 * (1 - t)).clip(0,255).round().astype(np.uint8)


def add_motion_blur(img, level: int):
    """Adds motion blur. `level: int` in {1,2,3,4}. This code is adapted from that of [imagecorruptions](https://github.com/bethgelab/imagecorruptions)."""

    img = img.copy()

    radius, sigma = [(5, 2), (10, 3), (15, 5), (15, 8)][level-1]
    
    angle = np.random.uniform(-90, 90)

    # optimal kernel width 1D
    width = radius * 2 + 1
    # motion blur kernel
    k = (np.exp(-np.arange(width)**2 / (2*(sigma**2)))) / (np.sqrt(2*np.pi)*sigma)  # gaussian
    kernel = k / np.sum(k)
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    hypot = math.hypot(point[0], point[1])

    blurred = np.zeros_like(img, dtype=np.float32)
    for i in range(width):
        dy = -math.ceil(((i*point[0]) / hypot) - 0.5)
        dx = -math.ceil(((i*point[1]) / hypot) - 0.5)
        if np.abs(dy) >= img.shape[0] or np.abs(dx) >= img.shape[1]:
            # simulated motion exceeded image borders
            break

        # shift
        if dx < 0:
            shifted = np.roll(img, shift=img.shape[1]+dx, axis=1)
            shifted[:,dx:] = shifted[:,dx-1:dx]
        elif dx > 0:
            shifted = np.roll(img, shift=dx, axis=1)
            shifted[:,:dx] = shifted[:,dx:dx+1]
        else:
            shifted = img

        if dy < 0:
            shifted = np.roll(shifted, shift=img.shape[0]+dy, axis=0)
            shifted[dy:,:] = shifted[dy-1:dy,:]
        elif dy > 0:
            shifted = np.roll(shifted, shift=dy, axis=0)
            shifted[:dy,:] = shifted[dy:dy+1,:]

        blurred = blurred + kernel[i] * shifted

    img = np.clip(blurred, 0, 255).round().astype(np.uint8)
    return img


def add_defocus_blur(img, level: int):
    """Adds defocus blur. `level: int` in {1,2,3,4}. This code is adapted from that of [imagecorruptions](https://github.com/bethgelab/imagecorruptions)."""

    img = img.copy()

    radius, alias_blur = [(2, 0.1), (3, 0.1), (4, 0.5), (6, 0.5)][level-1]

    # get kernel
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=np.float32)
    aliased_disk /= np.sum(aliased_disk)
    # supersample to antialias
    kernel = cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

    img = img / 255.0
    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(img[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))

    img = (np.clip(channels, 0, 1) * 255).round().astype(np.uint8)
    return img


def add_rain(img, level: int):
    """Adds rain. This code is adapted from that of [MiOIR](https://github.com/Xiangtaokong/MiOIR)."""

    img = img.copy()

    w = 3  # 粗细
    length = np.random.randint(20, 40)  # 长度（对角矩阵大小）
    angle = np.random.randint(-30, 30)  # 角度（逆时针为正）

    mean_value = [5, 60, 75, 85]
    value = max(np.random.normal(mean_value[level-1], 5), 3)
    noise = np.random.uniform(0, 256, img.shape[0:2])
    # 控制噪声水平，取浮点数，只保留最大的一部分作为噪声
    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0

    # 噪声做初次模糊
    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])

    noise = cv2.filter2D(noise, -1, k)

    # 这里由于对角阵自带45度的倾斜，逆时针为正，所以加了-45度的误差，保证开始为正
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))  # 生成对焦矩阵
    k = cv2.warpAffine(dig, trans, (length, length))  # 生成模糊核
    k = cv2.GaussianBlur(k, (w, w), 0)  # 高斯模糊这个旋转后的对角核，使得雨有宽度

    # k = k / length                         #是否归一化

    blurred = cv2.filter2D(noise, -1, k)  # 用刚刚得到的旋转后的核，进行滤波

    # 转换到0-255区间
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)

    rain = np.expand_dims(blurred, 2)
    rain = np.repeat(rain, 3, 2)

    img = img.astype('float32') + rain
    np.clip(img, 0, 255, out=img)

    return img.round().astype(np.uint8)


if __name__ == "__main__":
    input_path = Path("training_data/HQ/0004.png")
    img = cv2.imread(str(input_path))
    output_dir = Path("training_data/examples")
    shutil.copy(input_path, output_dir / "HQ.png")
    fns = [
        add_jpeg_comp_artifacts,
        add_motion_blur,
        add_defocus_blur,
        add_rain,
    ]
    
    for fn in fns:
        deg_dir = output_dir / fn.__name__
        if deg_dir.exists():
            shutil.rmtree(deg_dir)
        deg_dir.mkdir()
        for level in range(1, 5):
            new_img = fn(img, level)
            p = deg_dir / f"{level}.png"
            cv2.imwrite(str(p), new_img)

    lr_dir = output_dir / "lr"
    if lr_dir.exists():
        shutil.rmtree(lr_dir)
    lr_dir.mkdir()
    new_img = lr(img)
    p = lr_dir / f"4.png"
    cv2.imwrite(str(p), new_img)

    haze_dir = output_dir / "add_haze"
    if haze_dir.exists():
        shutil.rmtree(haze_dir)
    haze_dir.mkdir()
    for level in range(1, 5):
        new_img = add_haze(img, idx=input_path.stem, level=level)
        p = haze_dir / f"{level}.png"
        cv2.imwrite(str(p), new_img)

    darken_dir = output_dir / "darken"
    if darken_dir.exists():
        shutil.rmtree(darken_dir)
    darken_dir.mkdir()
    for darken_type in ["constant shift", "gamma correction", "linear mapping"]:
        deg_dir = darken_dir / darken_type
        deg_dir.mkdir()
        for level in range(1, 5):
            new_img = darken(img, level, darken_type)
            p = deg_dir / f"{level}.png"
            cv2.imwrite(str(p), new_img)

    noise_dir = output_dir / "add_noise"
    if noise_dir.exists():
        shutil.rmtree(noise_dir)
    noise_dir.mkdir()
    for noise_type in ["Gaussian", "Poisson"]:
        deg_dir = noise_dir / noise_type
        deg_dir.mkdir()
        for level in range(1, 5):
            new_img = add_noise(img, level, noise_type)
            p = deg_dir / f"{level}.png"
            cv2.imwrite(str(p), new_img)