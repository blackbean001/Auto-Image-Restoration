CUDA_VISIBLE_DEVICE=1 python src/imgres_test_submission.py \
   --submission-name 'imgres_test_jason' \
   --combining-function 'combiner' \
   --combiner-path '/home/jason/CLIP4Cir/models/combiner_trained_on_imgres_RN50x4_2025-09-05_12:30:03/saved_models/combiner_arithmetic.pt' \
   --projection-dim 2560 \
   --hidden-dim 5120 \
   --clip-model-name RN50x4 \
   --clip-model-path '/home/jason/CLIP4Cir/models/clip_finetuned_on_imgres_RN50x4_2025-09-05_10:48:31/saved_models/tuned_clip_arithmetic.pt' \
   --target-ratio 1.25 \
   --transform targetpad
