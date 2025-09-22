from pathlib import Path
import cv2
from tqdm import tqdm
import random

from add_single_degradation import *

random.seed(0)


def degrade(img, degradation, level, idx):
    # level of 1,2,3,4 means low, medium, high, very high
    assert level in {1, 2, 3, 4}
    router = {
        "low resolution": lr,
        "dark": darken,
        "noise": add_noise,
        "jpeg compression artifact": add_jpeg_comp_artifacts,
        "haze": add_haze,
        "motion blur": add_motion_blur,
        "defocus blur": add_defocus_blur,
        "rain": add_rain,
    }
    if degradation == "haze":
        return add_haze(img, level=level, idx=idx)
    return router[degradation](img, level=level)


base_dir = Path("training_data")
hq_dir = base_dir / "HQ"
lq_dir = base_dir / "LQ"
lq_dir.mkdir(parents=True, exist_ok=True)

all_degs = ["rain", "haze", "motion blur", "defocus blur", "dark",
            "noise", "jpeg compression artifact", "low resolution"]

hq_paths = sorted(list(hq_dir.glob("*")))
for n_degra in range(1, 4):
    nd_dir = lq_dir / f"d{n_degra}"
    nd_dir.mkdir()
    existing_imgs = set()
    for _ in tqdm(range(5000), desc=f"d{n_degra}", unit='img'):
        # randomly sample image, degradations and levels
        while True:
            hq_path = random.choice(hq_paths)
            degs = random.sample(all_degs, k=n_degra)
            degs.sort(key=lambda d: all_degs.index(d))
            levels = [
                random.randint(1, 4) for i in range(n_degra)
            ]
            degs_str = '+'.join(degs)
            level_str = ''.join(str(lv) for lv in levels)
            img_name = f"{hq_path.stem}@{degs_str}@{level_str}.png"
            if img_name not in existing_imgs:
                existing_imgs.add(img_name)
                break
        img = cv2.imread(str(hq_path))
        for deg, level in zip(degs, levels):
            img = degrade(img, deg, level=level, idx=hq_path.stem)
        save_path = nd_dir / img_name
        assert not save_path.exists()
        cv2.imwrite(str(save_path), img)
    assert len(existing_imgs) == 5000
    assert len(list(nd_dir.glob("*"))) == 5000