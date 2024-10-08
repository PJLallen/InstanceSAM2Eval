# Evaluation Study on SAM 2 for Class-agnostic Instance-level Segmentation
Code repository for the paper titled "[Evaluation Study on SAM 2 for Class-agnostic Instance-level Segmentation](https://arxiv.org/pdf/2409.02567)"

[[`Paper`](https://arxiv.org/pdf/2409.02567)] 

![SIS/CIS/SID](sample1.png)

![DIS](sample2.png)


## Get Started

### Install
1. Download the SAM and SAM2 code from the website. [[`SAM`](https://github.com/facebookresearch/segment-anything)] and [[`SAM2`](https://github.com/facebookresearch/segment-anything-2)]
2. Please follow the [[`SAM`](https://github.com/facebookresearch/segment-anything)] and [[`SAM2`](https://github.com/facebookresearch/segment-anything-2)] to install the enveriment. 
3. copy the code in SAM/notebooks into segment-anything/notebooks. Copy the code in SAM2/notebooks into segment-anythings/notebooks

### Datasets

**SIS**：
- **ILSO1K**: [Google Drive](https://drive.google.com/file/d/1mpGHOQtUHmZGEMC6KdC8iQYL0-hqzK5g/view?usp=sharing)
- **SOC**: [Google Drive](https://drive.google.com/file/d/1GYX5HAk3wwOqmgg2jaf6-6VgRzCdMSsL/view?usp=sharing)
- **SIS10K**[Baidu Disk](https://pan.baidu.com/s/1ZOQAj0Lhg1K4Vi3eS5Tw6w) Verification code: hust
- **SIP**: [Google Drive](https://drive.google.com/file/d/1ebNjyrS28vEXDGawxHxVFNNxz3XLBqrT/view?usp=drive_link)


**CIS**：
- **COD10K**: [Baidu](https://pan.baidu.com/s/1IPcPjdg1EJ-h9HPoU42nHA) (password:hust) / [Google](https://drive.google.com/file/d/1YGa3v-MiXy-3MMJDkidLXPt0KQwygt-Z/view?usp=sharing) / [Quark](https://pan.quark.cn/s/07ba3258b777); **Json files:** [Baidu](https://pan.baidu.com/s/1kRawj-hzBDycCkZZfQjFhg) (password:hust) / [Google](https://drive.google.com/drive/folders/1Yvz63C8c7LOHFRgm06viUM9XupARRPif?usp=sharing)
- **NC4K**: [Baidu](https://pan.baidu.com/s/1li4INx4klQ_j8ftODyw2Zg) (password:hust) / [Google](https://drive.google.com/file/d/1eK_oi-N4Rmo6IIxUNbYHBiNWuDDLGr_k/view?usp=sharing); **Json files:** [Baidu](https://pan.baidu.com/s/1DBPFtAL2iEjefwiqXE_GWA) (password:hust) / [Google](https://drive.google.com/drive/folders/1LyK7tl2QVZBFiNaWI_n0ZVa0QiwF2B8e?usp=sharing)


**SID**：
- **SOBA** dataset can be downloaded from [https://github.com/stevewongv/SSIS](https://github.com/stevewongv/SSIS)

**DIS**：
- **DIS5K** dataset can be downloaded from [https://github.com/xuebinqin/DIS](https://github.com/xuebinqin/DIS)

## Evaluation

**SIS/CIS/SID**:

1. Download the weights of SAM and SAM2 from the [website](https://github.com/facebookresearch/segment-anything-2).
2. Put weights into the **checkpoints/**
3. Use .sh files in **notebooks** to evaluate:

-->use SAM， auto model， SIS task as example：\\
modify the dataset root in **SAM/notebooks/[run_auto_saliency.sh](https://github.com/PJLallen/InstanceSAM2Eval/blob/main/SAM/notebooks/run_auto_saliency.sh)**

```bash
# SIS task / SAM / auto model
$ cd notebooks/
$ bash run_auto_saliency.sh
```
Note that, [sam/sam2]_[your mode]_shadow.sh means the SID task. For CIS task, you can change the dataset path and save_json path in [sam/sam2]_[your mode]_cos10k.py or [sam/sam2]_[your mode]_nc4k.py, then
```shell
python  [sam/sam2]_[your mode]_cos10k.py
# or 
python  [sam/sam2]_[your mode]_nc4k.py
```

SIS task use he AP70 metric instead of AP75.

After running the [sam/sam2]_[your mode]_saliency.sh, please change the path in [run_AP70.sh](https://github.com/PJLallen/InstanceSAM2Eval/blob/main/SAM/notebooks/run_AP70.sh), then 

```bash
$ bash run_AP70.sh
```

**DIS**:

1. To Get the predicted results of SAM.
- automatic prompts mode:
  
following the [[`SAM`](https://github.com/facebookresearch/segment-anything)], masks can be generated for images from the command line:
```shell
python scripts/amg.py --checkpoint <path/to/checkpoint> --model-type <model_type> --input <image_or_folder> --output <path/to/output>
```
then, selecting the most suitable foreground mask, use a maximum Intersection over Union (IoU)
```shell
cd DIS/script
python3 findMaxIoUMask.py 
```
- bounding box prompt mode:
```shell
cd DIS/SAM
python3 test_with_box_prompt_floder.py 
```

2. To Evaluate the predicted results.
```shell
cd DIS/metrics
python3 test_metrics.py 
python3 hce_metric_main.py
```

## Qualitative Results of SIS, CIS and SID
1. SAM2 [sam2_demo](https://drive.google.com/file/d/19fAYi0cr6V99T-LNU29itRwqUaQukvcB/view?usp=sharing)
2. SAM [sam_demo](https://drive.google.com/file/d/1sBoaD9JBD5vjPWPEGmUXuzmeV0gzjBTy/view?usp=sharing)

## Citation
If you find our work useful for your research or applications, please cite using this BibTeX:
```bibtex
@article{zhang2024evalsam2,
      title={Evaluation Study on SAM 2 for Class-agnostic Instance-level Segmentation}, 
      author={Zhang, Tiantian and Zhou, Zhangjun and Pei, Jialun},
      journal={arXiv preprint arXiv:2409.02567},
      year={2024},
      url={https://arxiv.org/abs/2409.02567} 
}
```

## Acknowledgement

Thanks for the efforts of the authors involved in the [Segment Anything](https://github.com/facebookresearch/segment-anything), [Segment Anything 2](https://github.com/facebookresearch/segment-anything-2) and [UnderwaterSAM2Eval](https://github.com/LiamLian0727/UnderwaterSAM2Eval).
