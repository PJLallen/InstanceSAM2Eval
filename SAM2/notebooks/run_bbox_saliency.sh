#!/bin/bash

# 参数列表
declare -a annotations=(
    "/research/d1/rshr/ttzhang/d2/saliency_dataset/ILSO1K/test300.json"
    "/research/d1/rshr/ttzhang/d2/saliency_dataset/SOC/SOC_test600.json"
    "/research/d1/rshr/ttzhang/d2/saliency_dataset/SIP/SIP.json"
)

declare -a images=(
    "/research/d1/rshr/ttzhang/d2/saliency_dataset/ILSO1K/test300"
    "/research/d1/rshr/ttzhang/d2/saliency_dataset/SOC/test/images"
    "/research/d1/rshr/ttzhang/d2/saliency_dataset/SIP/RGB"
)

declare -a model_types=(
    "tiny"
    "base+"
    "large"
)
declare -a model_cfg=(
    "sam2_hiera_t.yaml"
    "sam2_hiera_b+.yaml"
    "sam2_hiera_l.yaml"
)

declare -a checkpoints=(
    "../checkpoints/sam2_hiera_tiny.pt"
    "../checkpoints/sam2_hiera_base_plus.pt"
    "../checkpoints/sam2_hiera_large.pt"
)

log_file="sam2_bbox_saliency1.log"

# 清空日志文件
> "$log_file"

# 外层循环
for ((i=0; i<${#annotations[@]}; i++)); do
    # 获取annotation的最后一部分
    annotation_name=$(basename "${annotations[i]}" .json)
    
    # 内层循环
    for ((j=0; j<${#model_types[@]}; j++)); do
        output_file="sam2_bbox_${model_types[j]}_${annotation_name}.json"
        {
            echo "Running configuration: annotation ${i+1}, model ${model_types[j]}"
            python sam2_bbox_eval_saliency.py \
                --annotation_path "${annotations[i]}" \
                --image_path "${images[i]}" \
                --model_cfg "${model_cfg[j]}" \
                --device "cuda" \
                --sam2_checkpoint "${checkpoints[j]}" \
                --save_json "$output_file"
            echo "Finished configuration: annotation ${i+1}, model ${model_types[j]}"
        } >> "$log_file" 2>&1
    done
done