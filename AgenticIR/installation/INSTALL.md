# Installation

### Set up the environment

```bash
conda create -n agenticir python=3.10
conda activate agenticir
pip install -r installation/requirements.txt
```

### Deploy IR models

We employ the following models as single-degradation restoration tools: [DiffBIR](https://github.com/XPixelGroup/DiffBIR), [X-Restormer](https://github.com/Andrew0613/X-Restormer), [SwinIR](https://github.com/JingyunLiang/SwinIR), [HAT](https://github.com/XPixelGroup/HAT), [MPRNet](https://github.com/swz30/MPRNet), [MAXIM](https://github.com/google-research/maxim), [Restormer](https://github.com/swz30/Restormer), [DRBNet](https://github.com/lingyanruan/DRBNet), [IFAN](https://github.com/codeslake/IFAN), [RIDCP](https://github.com/RQ-Wu/RIDCP_dehazing), [DehazeFormer](https://github.com/IDKiro/DehazeFormer), and [FBCNN](https://github.com/jiaxi-jiang/FBCNN). Thanks for these awesome works.

+ Run `sh installation/deploy_tools.sh` to prepare the code, which is adapted from the official repos linked above.
+ Set up their respective environments according to the official repos.
    + Note: The environment name has [a role in our framework](https://github.com/Kaiwen-Zhu/AgenticIR/blob/main/executor/tool.py#L73). It is recommended to use the names listed below. Otherwise, you may need to modify the scripts in `executor`.
        <details>

        <summary>Recommended environment names</summary>

        + DiffBIR: `diffbir`
        + X-Restormer: `xrestormer`
        + SwinIR: `swinir`
        + HAT: `hat`
        + MPRNet: `mprnet`
        + MAXIM: `maxim`
        + Restormer: `restormer`
        + DRBNet: `drbnet`
        + IFAN: `ifan`
        + RIDCP: `ridcp`
        + DehazeFormer: `dehazeformer`
        + FBCNN: `fbcnn`

        </details>

+ Download the weights. You may need to modify the paths in scripts in `executor`. A tutorial will be given if necessary.
+ Run `python -m test_tool.test_tool` to check whether all tools work properly.

PS: In our implementation, we use DiffBIR of the [`7bd5675`](https://github.com/XPixelGroup/DiffBIR/commit/7bd5675823c157b9afdd479b59a2bf0a8954ce11) commit version. After that, DiffBIR has undergone an overhaul, which may cause compatibility issues. It is recommended to use the code and weights of this version (`installation/deploy_tools.sh` has already checked out this version). Otherwise, you may need to customize an `inference.py` script in the `DiffBIR` directory following the logic of the tool call in our framework.

### Deploy [DepictQA](https://github.com/XPixelGroup/DepictQA)
+ Run `sh installation/deploy_depictqa.sh` to prepare the code, which is adapted from the official repo linked above.
+ Set up the environment according to the official repo.
+ Download the weights.
    + Download the pre-trained ViT from [this link](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) and put it in `DepictQA/weights/`.
    + Download the pre-trained Vicuna from [this link](https://huggingface.co/lmsys/vicuna-7b-v1.5/tree/main) and put it in `DepictQA/weights/`.
    + Download the delta weights of DepictQA-Wild from [this link](https://huggingface.co/zhiyuanyou/DepictQA2-Abstractor-DQ495K/blob/main/ckpt.pt), rename it to `DQ495K_Abstractor.pt`, and put it in `DepictQA/weights/delta/`.
    + Download the delta weights fine-tuned from DepictQA-Wild from [this link](https://drive.google.com/file/d/1o-PN1iXctWl62Tdb8fZs1eD1Ehv6HBMh/view?usp=drive_link) and put it in `DepictQA/weights/delta/`.

    The structure of `DepictQA/weights` should look like this:
    ```
    DepictQA/weights/
    ├── ViT-L-14.pt
    ├── vicuna-7b-v1.5/
    │   └── ...
    └── delta/
        ├── DQ495K_Abstractor.pt
        └── degra_eval.pt
    ```
