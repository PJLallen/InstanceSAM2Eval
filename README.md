# InstanceSAM2Eval
Code repository for our paper titled "Evaluation Study on SAM 2 for Class-agnostic Instance-level Segmentation"

[[`Paper`](https://arxiv.org/pdf/2409.02567)] 


## Get Started

### install
1. Download the SAM and SAM2 code from the website. [[`SAM`](https://github.com/facebookresearch/segment-anything)] and [[`SAM2`](https://github.com/facebookresearch/segment-anything-2)]
2. Please follow the [[`SAM`](https://github.com/facebookresearch/segment-anything)] and [[`SAM2`](https://github.com/facebookresearch/segment-anything-2)] to install the enveriment. 
3. copy the code in SAM/notebooks into segment-anything/notebooks. Copy the code in SAM2/notebooks into segment-anythings/notebooks

### Dataset

SIS：
- **ILSO1K**: [Google Drive](https://drive.google.com/file/d/1mpGHOQtUHmZGEMC6KdC8iQYL0-hqzK5g/view?usp=sharing)
- **SOC**: [Google Drive](https://drive.google.com/file/d/1GYX5HAk3wwOqmgg2jaf6-6VgRzCdMSsL/view?usp=sharing)
- **SIS10K**[Baidu Disk](https://pan.baidu.com/s/1ZOQAj0Lhg1K4Vi3eS5Tw6w) Verification code: hust
- **SIP**: [Google Drive](https://drive.google.com/file/d/1ebNjyrS28vEXDGawxHxVFNNxz3XLBqrT/view?usp=drive_link)


CIS：
- **COD10K**: [Baidu](https://pan.baidu.com/s/1IPcPjdg1EJ-h9HPoU42nHA) (password:hust) / [Google](https://drive.google.com/file/d/1YGa3v-MiXy-3MMJDkidLXPt0KQwygt-Z/view?usp=sharing) / [Quark](https://pan.quark.cn/s/07ba3258b777); **Json files:** [Baidu](https://pan.baidu.com/s/1kRawj-hzBDycCkZZfQjFhg) (password:hust) / [Google](https://drive.google.com/drive/folders/1Yvz63C8c7LOHFRgm06viUM9XupARRPif?usp=sharing)
- **NC4K**: [Baidu](https://pan.baidu.com/s/1li4INx4klQ_j8ftODyw2Zg) (password:hust) / [Google](https://drive.google.com/file/d/1eK_oi-N4Rmo6IIxUNbYHBiNWuDDLGr_k/view?usp=sharing); **Json files:** [Baidu](https://pan.baidu.com/s/1DBPFtAL2iEjefwiqXE_GWA) (password:hust) / [Google](https://drive.google.com/drive/folders/1LyK7tl2QVZBFiNaWI_n0ZVa0QiwF2B8e?usp=sharing)


SID：
SOBA dataset can be downloaded from [here](https://github.com/stevewongv/SSIS)

DIS：
DIS5K dataset can be downloaded from [here](https://github.com/xuebinqin/DIS)

### Evaluation
SIS/CIS/SID:
1. Download the weights of SAM and SAM2 from the website
2. Put the weight into the **checkpoints/**
3. Use the SIS as example:
-> SAM -> auto model \\

modify the dataset root in **SAM/notebooks/[run_auto_saliency.sh](https://github.com/PJLallen/InstanceSAM2Eval/blob/main/SAM/notebooks/run_auto_saliency.sh)**

```bash
# SIS task / SAM / auto model
$ cd notebooks/
$ bash run_auto_saliency.sh
```

DIS:
- To Get the predicted results of SAM.
  automatic prompts mode:
following the [[`SAM`](https://github.com/facebookresearch/segment-anything)]. masks can be generated for images from the command line:
```shell
python scripts/amg.py --checkpoint <path/to/checkpoint> --model-type <model_type> --input <image_or_folder> --output <path/to/output>
```
then,selecting the most suitable foreground mask, use a maximum Intersection over Union (IoU)
```shell
cd DIS/script
python3 test_metrics.py 
```

  bounding box prompt mode:

- To Evaluate the predicted results.
```shell
cd DIS/metrics
python3 test_metrics.py 
python3 hce_metric_main.py
```

## Qualitative Results of SIS, CIS and SID task
1. SAM2 [sam2_demo](https://drive.google.com/file/d/19fAYi0cr6V99T-LNU29itRwqUaQukvcB/view?usp=sharing)
2. SAM [sam_demo](https://drive.google.com/file/d/1sBoaD9JBD5vjPWPEGmUXuzmeV0gzjBTy/view?usp=sharing)

## Citation
If you find our work useful for your research or applications, please cite using this BibTeX:
```bibtex
@misc{zhang2024evaluationstudysam2,
      title={Evaluation Study on SAM 2 for Class-agnostic Instance-level Segmentation}, 
      author={Tiantian Zhang and Zhangjun Zhou and Jialun Pei},
      year={2024},
      eprint={2409.02567},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.02567}, 
}
```

## Acknowledgement

Thanks for the efforts of the authors involved in the [Segment Anything](https://github.com/facebookresearch/segment-anything), [Segment Anything 2](https://github.com/facebookresearch/segment-anything-2) and [UnderwaterSAM2Eval](https://github.com/LiamLian0727/UnderwaterSAM2Eval)
