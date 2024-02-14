export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./output"

accelerate launch main.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --output_format=both \
  --instance_prompt="a photo of sks person" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --max_train_steps=200 \
  --seed="404" \
  --mixed_precision=no \
  --num_class_images=200 \
  --src_dir="./data/src" \
  --targ_dir="./data/targ"