#!/bin/bash

# 文件名列表
json_files=(
"saliency/sam2_auto_tiny_SIP.json"
"saliency/sam2_auto_tiny_SIS10K_test_m.json"
"saliency/sam2_auto_tiny_SOC_test600.json"
"saliency/sam2_auto_tiny_test300.json"
"saliency/sam2_auto_base+_SIP.json"
"saliency/sam2_auto_base+_SIS10K_test_m.json"
"saliency/sam2_auto_base+_SOC_test600.json"
"saliency/sam2_auto_base+_test300.json"
"saliency/sam2_auto_large_SIP.json"
"saliency/sam2_auto_large_SIS10K_test_m.json"
"saliency/sam2_auto_large_SOC_test600.json"
"saliency/sam2_auto_large_test300.json"
"saliency/sam2_bbox_tiny_SIP.json"
"saliency/sam2_bbox_tiny_SIS10K_test_m.json"
"saliency/sam2_bbox_tiny_SOC_test600.json"
"saliency/sam2_bbox_tiny_test300.json"
"saliency/sam2_bbox_base+_SIP.json"
"saliency/sam2_bbox_base+_SIS10K_test_m.json"
"saliency/sam2_bbox_base+_SOC_test600.json"
"saliency/sam2_bbox_base+_test300.json"
"saliency/sam2_bbox_large_SIP.json"
"saliency/sam2_bbox_large_SIS10K_test_m.json"
"saliency/sam2_bbox_large_SOC_test600.json"
"saliency/sam2_bbox_large_test300.json"
)




annotation_paths=(
    "/research/d1/rshr/ttzhang/d2/saliency_dataset/SIP/SIP.json"
    "/research/d1/rshr/ttzhang/d2/saliency_dataset/SIS10K/test/SIS10K_test_m.json"
    "/research/d1/rshr/ttzhang/d2/saliency_dataset/SOC/SOC_test600.json"
    "/research/d1/rshr/ttzhang/d2/saliency_dataset/ILSO1K/test300.json"
)

# 日志文件
log_file="evaluation_ap70_sam2.log"

# 清空日志文件
> "$log_file"

# 循环遍历文件名
for i in "${!json_files[@]}"; do
    json_file="${json_files[$i]}"
    # 根据索引计算对应的 annotation 文件
    annotation_file="${annotation_paths[$((i % ${#annotation_paths[@]}))]}"
    echo "Evaluating $json_file with $annotation_file" | tee -a "$log_file"
    python eva_AP70.py --json_file "$json_file" --annotation_path "$annotation_file" 2>&1 | tee -a "$log_file"
done