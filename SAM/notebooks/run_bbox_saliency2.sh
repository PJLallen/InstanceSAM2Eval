#!/bin/bash

declare -a annotations=(
    "/research/d1/rshr/ttzhang/d2/saliency_dataset/SIS10K/test/SIS10K_test_m.json"
)

declare -a images=(
    "/research/d1/rshr/ttzhang/d2/saliency_dataset/SIS10K/test/images"
)

declare -a model_types=(
    "vit_b"
    "vit_l"
    "vit_h"
)

declare -a checkpoints=(
    "../checkpoints/sam_vit_b.pth"
    "../checkpoints/sam_vit_l.pth"
    "../checkpoints/sam_vit_h.pth"
)

log_file="sam_bbox_saliency2.log"

# 清空日志文件
> "$log_file"

# 外层循环
for ((i=0; i<${#annotations[@]}; i++)); do
    # 获取annotation的最后一部分
    annotation_name=$(basename "${annotations[i]}" .json)
    
    # 内层循环
    for ((j=0; j<${#model_types[@]}; j++)); do
        output_file="sam_bbox_${model_types[j]}_${annotation_name}.json"
        {
            echo "Running configuration: annotation ${i+1}, model ${model_types[j]}"
            python sam_bbox_saliency.py \
                --annotation_path "${annotations[i]}" \
                --image_path "${images[i]}" \
                --model_type "${model_types[j]}" \
                --device "cuda" \
                --sam_checkpoint "${checkpoints[j]}" \
                --save_json "$output_file"
            echo "Finished configuration: annotation ${i+1}, model ${model_types[j]}"
        } >> "$log_file" 2>&1
    done
done