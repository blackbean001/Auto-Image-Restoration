To fine-tune DepictQA, follow these steps:
+ Data preparation
    + Put the high-quality images in `DepictQA/experiments/agenticir/training_data/HQ` and the corresponding depth map in `DepictQA/experiments/agenticir/training_data/depth`, the structure of the directory `training_data` should look like this:
        ```
        training_data/
        ├── HQ/
        │   ├── 0001.png
        │   └── ...
        └── depth/
            ├── 0001
            │   └── predict_depth.mat
            └── ...
        ```
        In our implementation, we use the data from [MiOIR](https://github.com/Xiangtaokong/MiOIR?tab=readme-ov-file#step1-download-the-training-data).
    + `cd DepictQA/experiments/agenticir`
    + Run `python synthesize_lq.py` to synthesize the low-quality images.
    + Run `python build_meta.py` to synthesize the training data.
+ In `DepictQA/experiments/agenticir/config_train.yaml`, set the weight path of ViT (line 33), Vicuna (line 42), and the [pre-trained DepictQA](https://huggingface.co/zhiyuanyou/DepictQA2-Abstractor-DQ495K/blob/main/ckpt.pt) (line 49).
+ Run `sh train.sh ids_of_gpus` to fine-tune the model. The weights will be saved to `DepictQA/experiments/agenticir/ckpt/ckpt.pt`.