from pathlib import Path
import json
from tqdm import tqdm


root_data_dir = Path("training_data").resolve()
lq_train_dir = root_data_dir / "LQ"

all_degradation = ["noise", "motion blur", "defocus blur",
                   "rain", "haze", "dark", "jpeg compression artifact"]
all_levels = ["very low", "low", "medium", "high", "very high"]


def sorted_glob(dir: Path, pattern: str = '*'):
    assert dir.is_dir()
    return sorted(list(dir.glob(pattern)))


def build_one_entry(img_path: Path, degradation: str, id: str, level: str):
    return {
        "id": id,
        "image_ref": None,
        "image_A": str(img_path),
        "image_B": None,
        "task_type": "quality_single_A_noref",
        "conversations": [
            {
                "from": "human",
                "value": f"What's the severity of {degradation} in this image? Answer the question using a single word or phrase in the followings: very low, low, medium, high, very high."
            },
            {
                "from": "gpt",
                "value": level
            }
        ]
    }


meta_dir = root_data_dir / "meta"
meta_dir.mkdir(exist_ok=True)

all_meta_paths = []
for nd_dir in sorted_glob(lq_train_dir):
    imgs = sorted_glob(nd_dir)
    for i in range(5):
        split = imgs[1000*i:1000*(i+1)]
        meta_path = meta_dir / f"{nd_dir.stem}-{i}.json"
        all_meta_paths.append(str(meta_path))
        meta_lst = []
        for img_path in split:
            img_id, degs, levels = img_path.stem.split('@')
            deg_lst = degs.split('+')
            level_lst = [int(lv) for lv in levels]
            for degradation in all_degradation:
                id = f"{img_path.stem}@{degradation}"
                if degradation in deg_lst:
                    this_idx = deg_lst.index(degradation)
                    level = all_levels[level_lst[this_idx]]
                else:
                    level = "very low"
                meta_lst.append(build_one_entry(img_path, degradation, id, level))
        assert len(meta_lst) == 7000
        with meta_path.open('w') as f:
            json.dump(meta_lst, f, indent=2)

assert len(list(meta_dir.glob("*.json"))) == 15
assert len(all_meta_paths) == 15
print('\n'.join(all_meta_paths))
