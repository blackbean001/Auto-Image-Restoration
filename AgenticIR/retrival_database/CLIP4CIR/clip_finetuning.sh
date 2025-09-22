CUDA_VISIBLE_DEVICE=1
python src/clip_fine_tune.py \
   --dataset 'ImgRes' \
   --api-key '038UtfPXqcfLarzK3KlAKxMuN' \
   --workspace 'clip4cir-imgres' \
   --experiment-name 'first_demo' \
   --num-epochs 80 \
   --clip-model-name RN50x4 \
   --encoder both \
   --learning-rate 2e-6 \
   --batch-size 24 \
   --transform targetpad \
   --target-ratio 1.25  \
   --save-training \
   --save-best \
   --validation-frequency 1 
