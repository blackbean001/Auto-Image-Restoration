CUDA_VISIBLE_DEVICE=0
python -m src.insert_emb_to_postgresql \
  --imgs-dir "/home/jason/AgenticIR/output"  \
  --combiner-path "/home/jason/CLIP4Cir/models/combiner_trained_on_imgres_RN50x4_2025-09-05_12:30:03/saved_models/combiner_arithmetic.pt"  \
  --combining-function 'combiner' \
  --clip-model-path "/home/jason/CLIP4Cir/models/clip_finetuned_on_imgres_RN50x4_2025-09-05_10:48:31/saved_models/tuned_clip_arithmetic.pt"  \
  --projection-dim 2560 \
  --hidden-dim 5120 \
  --transform "targetpad"  \
  --target-ratio 1.25  \
