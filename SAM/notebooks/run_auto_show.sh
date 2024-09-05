#!/bin/bash

# Define arrays with your different input values
ANNOTATION_PATHS=(
    "/research/d1/rshr/ttzhang/d2/dataset/json/test2026.json"
    "/research/d1/rshr/ttzhang/d2/dataset/json/nc4k_test.json"
    "/research/d1/rshr/ttzhang/d2/saliency_dataset/ILSO1K/test300.json"
    "/research/d1/rshr/ttzhang/d2/saliency_dataset/SIS10K/test/SIS10K_test_m.json"
    "/research/d1/rshr/ttzhang/d2/saliency_dataset/SOC/SOC_test600.json"
    "/research/d1/rshr/ttzhang/d2/saliency_dataset/SIP/SIP.json"
    "/research/d1/rshr/ttzhang/d2/dataset/shadow/SOBA/annotations/SOBA_challenge_v2.json"
    "/research/d1/rshr/ttzhang/d2/dataset/shadow/SOBA/annotations/SOBA_val_v2.json"
)

IMAGE_PATHS=(
    "/research/d1/rshr/ttzhang/d2/dataset/COD10K/Test_Image_CAM"
    "/research/d1/rshr/ttzhang/d2/dataset/NC4K/test/image"
    "/research/d1/rshr/ttzhang/d2/saliency_dataset/ILSO1K/test300"
    "/research/d1/rshr/ttzhang/d2/saliency_dataset/SIS10K/test/images"
    "/research/d1/rshr/ttzhang/d2/saliency_dataset/SOC/test/images"
    "/research/d1/rshr/ttzhang/d2/saliency_dataset/SIP/RGB"
    "/research/d1/rshr/ttzhang/d2/dataset/shadow/SOBA/SOBA"
    "/research/d1/rshr/ttzhang/d2/dataset/shadow/SOBA/SOBA"
)

SAVE_PATHS=(
    "/research/d1/rshr/ttzhang/segment-anything/demo_results2/cod10k_auto_huge"
    "/research/d1/rshr/ttzhang/segment-anything/demo_results2/NC4K_auto_huge"
    "/research/d1/rshr/ttzhang/segment-anything/demo_results2/ILS01K_auto_huge"
    "/research/d1/rshr/ttzhang/segment-anything/demo_results2/SIS10K_auto_huge"
    "/research/d1/rshr/ttzhang/segment-anything/demo_results2/SOC_auto_huge"
    "/research/d1/rshr/ttzhang/segment-anything/demo_results2/SIP_auto_large"
    "/research/d1/rshr/ttzhang/segment-anything/demo_results2/SOBA_challenge_auto_huge"
    "/research/d1/rshr/ttzhang/segment-anything/demo_results2/SOBA_val_auto_large"
)

SAM_CHECKPOINTS=(
    "../checkpoints/sam_vit_h.pth"
    "../checkpoints/sam_vit_h.pth"
    "../checkpoints/sam_vit_h.pth"
    "../checkpoints/sam_vit_h.pth"
    "../checkpoints/sam_vit_h.pth"
    "../checkpoints/sam_vit_l.pth"
    "../checkpoints/sam_vit_h.pth"
    "../checkpoints/sam_vit_l.pth"

)

MODEL_CFGS=(
    "vit_h"
    "vit_h"
    "vit_h"
    "vit_h"
    "vit_h"
    "vit_l"
    "vit_h"
    "vit_l"

)



# Loop through the arrays and execute the Python script with each set of arguments
for i in "${!ANNOTATION_PATHS[@]}"; do
    python sam_auto_show.py \
        --annotation_path "${ANNOTATION_PATHS[i]}" \
        --image_path "${IMAGE_PATHS[i]}" \
        --save_path "${SAVE_PATHS[i]}" \
        --sam_checkpoint "${SAM_CHECKPOINTS[i]}" \
        --model_cfg "${MODEL_CFGS[i]}"
done