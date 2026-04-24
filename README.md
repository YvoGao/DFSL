# Few-Shot Learning via Drifting

Official implementation of the paper [Few-Shot Learning via Drifting].



## Table of Contents

- [Installation](#️-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Project Structure](#-project-structure)
- [Citation](#citation)

## ⚙️ Installation

### Prerequisites

- Python 3.8.20
- Pytorch 2.3.0
- CUDA 12.1 

### Environment Setup




```bash
git clone git@github.com:YvoGao/DFSL.git
cd DFSL
```


```bash
conda create -n fma python=3.8.20
conda activate fma
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install git+https://github.com/openai/CLIP.git
pip install scipy==1.10.1
```

## 📁 Dataset Preparation


### Setup Instructions

1. Create a data directory:
```bash
mkdir -p data
```

2. Follow the detailed instructions in [DATASETS.md](DATASETS.md) to download and organize each dataset.

3. The expected directory structure:
```
.data/
├── oxford_pets/
├── eurosat/
├── ucf101/
├── sun397/
├── caltech-101/
├── dtd/
├── fgvc_aircraft/
├── food-101/
├── oxford_flowers/
├── stanford_cars/
└── imagenet/
```

4. For feature extractors like coop, cocoop, adapter and lora, first download according pre-trained checkpoints:
```
wget https://github.com/HKUST-LongGroup/FMA/releases/download/FMA_PEFT/FMA_PEFT.tar.gz
```
Then unpack it ```tar -xzvf FMA_PEFT.tar.gz``` , the expected directory structure:
```
.checkpoints/
├── eurosat/
│   └── coop_vit_b16_16s.pth
│   └── cocoop_vit_b16_16s.pth
│   └── adapter_vit_b16_16s.pth
|   └── lora_vit_b16_16s.pth
├── oxford_pets/
├── ...
```
## 🚀 Training

###  Basic Training

Train a model with default configuration (EuroSAT, 16-shot, CLIP ViT-B/16):

```bash
python train.py
```
### Configuration File

You can also modify the default configuration in `config.py`:

```python
class DefaultConfig:
    def __init__(self):
        self.epochs = 200
        self.batch_size = 32
        self.lr = 2e-4
        self.clip_type = 'ViT-B/16'  # CLIP backbone
        self.dataset = 'EuroSAT'
        self.num_shots = 16
        self.blocks = 12  # Number of residual blocks
        # ... other parameters
```


### Custom Configuration

In addition to edit the ```config.py```, you can also customize the training by specifying command-line arguments:

```bash
# Train on a specific dataset
python train.py --dataset OxfordPets

# Specify number of shots
python train.py --dataset EuroSAT --num_shots 8

# Choose feature extractor
python train.py --feature_extractor coop

# Combine multiple options
python train.py --dataset UCF101 --num_shots 4 --feature_extractor clip

# other options, like seed, epochs, batch size, blocks, etc.
python train.py --dataset UCF101 --num_shots 4 --feature_extractor lora --seed 1 --epochs 300 --batch_size 64 --blocks 5
```

This arguments will override the default setting( in ```DefaultConfig```).

### Available Arguments

- `--dataset`: Dataset name (default: `EuroSAT`)
  - Options: `OxfordPets`, `EuroSAT`, `UCF101`, `SUN397`, `Caltech101`, `DescribableTextures`, `FGVCAircraft`, `Food101`, `OxfordFlowers`, `StanfordCars`, `ImageNet`
  
- `--num_shots`: Number of shots for few-shot learning (default: `16`)
  - Options: `1`, `2`, `4`, `8`, `16`
  
- `--feature_extractor`: Feature extractor type (default: `coop`)
  - Options: `clip`, `coop`, `cocoop`, `adapter`, `lora`

- `--seed`: Random seed for reproducibility (default: `1`)

- `--gamma`: Stochastic noise level for feature interpolation (default: `0`)

- `--epochs`: Number of training epochs (default: `200`)

- `--batch_size`: Batch size for training (default: `32`)

- `--blocks`: Number of residual blocks in velocity network (default: `12`)


### Training Output

Training creates a timestamped checkpoint directory:
```
checkpoints/
└── 010857/
    ├── config.json      # Training configuration
    ├── model.pth        # Trained model weights
    └── log.txt          # Training logs
```

## 🧪 Evaluation

### Test a Trained Model

To evaluate a trained model, change the path of the checkpoint:

```bash
python test.py
```


This will:
- Load the saved configuration and model weights
- Evaluate the model on the test set with different inference steps (0-10)
- Report accuracy for each number of steps


## 🤖 Project Structure

```
.
├── config.py                  # Configuration and hyperparameters
├── train.py                   # Training script
├── test.py                    # Evaluation script
├── models/
│   ├── fm.py                  # Flow matching network
│   ├── utils.py               # Tool definition and class.
│   ├── feature_extractor.py   # Feature extraction interface
│   ├── coop_extractor.py      # CoOp extraction interface
│   ├── cocoop_extractor.py    # CoCoOp extraction interface
│   ├── lora_extractor.py      # CLIP-LoRA extraction interface
│   ├── adapter_extractor.py   # CLIP-Adapter extraction interface
│   └── clip_extractor.py      # CLIP feature extractor
├── datasets/
│   ├── __init__.py           # Dataset registry
│   ├── eurosat.py            # EuroSAT dataset
│   ├── oxford_pets.py        # Oxford Pets dataset
│   └── ...                   # Other datasets
├── checkpoints/              # Saved models 
├── data/                     # Datasets 
├── DATASETS.md              # Dataset preparation guide
└── README.md                # This file
```


## Acknowledgments

- [CLIP](https://github.com/openai/CLIP) for pre-trained vision-language models
- [CoOp](https://github.com/KaiyangZhou/CoOp) for dataset preparation scripts


## Citation
```
will soon
```