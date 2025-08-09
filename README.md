# DISSect: Differential-informed Sample Selection Accelerates Multimodal Contrastive Learning

<div align="center">   <a href="https://arxiv.org/abs/2507.12998">     <img src="https://img.shields.io/badge/arXiv-2507.12998-b31b1b" alt="arXiv">   </a>  
  <a href="https://github.com/MediaBrain-SJTU/DISSect">     <img src="https://img.shields.io/badge/GitHub-DISSect-brightgreen" alt="GitHub">   </a> </div>

by Zihua Zhao, Feng Hong, Mengxi Chen, Pengyi Chen, Benyuan Liu, Jiangchao Yao, Ya Zhang, Yanfeng Wang at Cooperative Medianet Innovation Center at Shanghai Jiao Tong University, School of AI at Shanghai Jiao Tong University and Shanghai AI Laboratory. 

This paper has been accepted by International Conference on Computer Vision (ICCV) 2025. This repo is the official Pytorch implementation of DISSect.

⚠️ This repository is being organized and updated continuously. Please note that this version is not the final release.

## 🚀 Quick Start

### Installation

Code is implemented based on original code provided by Open CLIP from https://github.com/mlfoundations/open_clip, which offers the standard code for retrieval framework, data loader and evaluation metrics. Besides, create the environment for running our code:

1. Clone the repository:

```bash
git clone MediaBrain-SJTU/DISSect
cd DISSect
```

2. Install dependencies:

```bash
conda create --name DISSect python=3.9
conda activate DISSect
pip install -r requirements.txt
```

### Configuration

Before running the training, you need to configure the following parameters:

1. **Dataset Paths**: Update the data paths in your training script. Note that we are using the public available **webdataset** form of CC3M and CC12M provided by https://huggingface.co/pixparse for efficient data loading. You should also process your own data into webdataset form for customization.

```bash
--train-data 'path/to/your/cc3m-train-{0000..0575}.tar'
--val-data 'path/to/your/cc3m-validation-{0000..0015}.tar'
```

2. **Dataset Size**: Specify the number of samples in your dataset:

```bash
--train-num-samples 2905954  # For CC3M
--train-num-samples 12423374 # For CC12M  
--train-num-samples 14681591 # For YFCC15M
```

### Training

Run training on CC3M dataset with your preferred selection strategy:

```bash
bash train_cc3m.sh
```

Or customize the parameters:

```bash
torchrun --nproc_per_node 4 src/main.py \
    --train-data 'path/to/your/train-data.tar' \
    --val-data 'path/to/your/val-data.tar' \
    --train-num-samples <your-dataset-size> \
    --select \
    --select-strategy 'warmup_base_sampling' \
    --select-rate 0.2 \
    --epochs 40
```

## 📊 Supported Selection Strategies

- `small_loss`: Select samples with smallest loss
- `big_loss`: Select samples with largest loss  
- `clipscore`: Select samples with highest CLIP score
- `random`: Random selection
- `historical_base_sampling`: Momentum version of DISSect
- `warmup_base_sampling`: Warm-up version of DISSect

## 🔧 Key Parameters

- `--select`: Enable data selection
- `--select-strategy`: Choose selection strategy
- `--select-rate`: Selection rate (0.0-1.0)
- `--warmup-point`: Number of warmup epochs

## 📝 Notes

- GPU memory usage depends on batch size and model size. We run DISSect on 8 A100 GPUs during experiments.
- The extra forward propagation is an inherent overhead of the online batch selection paradigm and can be accelerated through further low-level optimizations. The reported wall-clock time in Table 6 in the main paper only reflects our algorithm's core efficiency.

## 🤝 Citation

If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation:

```bibtex
@article{zhao2025differential,
  title={Differential-informed Sample Selection Accelerates Multimodal Contrastive Learning},
  author={Zhao, Zihua and Hong, Feng and Chen, Mengxi and Chen, Pengyi and Liu, Benyuan and Yao, Jiangchao and Zhang, Ya and Wang, Yanfeng},
  booktitle={Proceedings of the IEEE/CVF Conference on International Conference on Computer Vision},
  year={2025}
}
```
