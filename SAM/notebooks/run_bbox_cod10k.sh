#!/bin/bash

# 参数列表
declare -a annotations=(
    "/research/d1/rshr/ttzhang/d2/dataset/json/test2026.json"
    "/research/d1/rshr/ttzhang/d2/dataset/json/test2026.json"
    "/research/d1/rshr/ttzhang/d2/dataset/json/test2026.json"
)

declare -a images=(
    "/research/d1/rshr/ttzhang/d2/dataset/COD10K/Test_Image_CAM"
    "/research/d1/rshr/ttzhang/d2/dataset/COD10K/Test_Image_CAM"
    "/research/d1/rshr/ttzhang/d2/dataset/COD10K/Test_Image_CAM"
)
declare -a model_type=(
    "vit_b"
    "vit_l"
    "vit_h"
)
declare -a checkpoints=(
    "../checkpoints/sam_vit_b.pth"
    "../checkpoints/sam_vit_l.pth"
    "../checkpoints/sam_vit_h.pth"
)

declare -a outputs=(
    "sam_bbox_cod10k_base.json"
    "sam_bbox_cod10k_large.json"
    "sam_bbox_cod10k_huge.json"
)

# 循环执行每组参数
for i in "${!annotations[@]}"; do
    python sam_bbox_cod10k.py \
        --annotation_path "${annotations[i]}" \
        --image_path "${images[i]}" \
        --model_type "${model_type[i]}" \
        --device "cuda" \
        --sam_checkpoint "${checkpoints[i]}" \
        --save_json "${outputs[i]}"
done