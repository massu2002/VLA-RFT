# VLA-RFT: Vision-Language-Action Models with Reinforcement Fine-Tuning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![arXiv 2510.00406](https://img.shields.io/badge/arXiv-2510.00406-b31b1b?logo=arXiv&logoColor=white)](https://arxiv.org/abs/2510.00406)
[![Project Page](https://img.shields.io/badge/Project%20Page-vla--rft.github.io-6f42c1?logo=github&logoColor=white)](https://vla-rft.github.io/)

<div id="top" align="center">
<p align="center">
<img src=image/Figure1.png width=90% />
</p>
</div>

<div id="top" align="center">
<p align="center">
<img src=image/Figure2.png width=90% />
</p>
</div>

Vision-Language-Action (VLA) models enable embodied decision-making but rely heavily on imitation learning, leading to compounding errors and poor robustness under distribution shift. Reinforcement learning (RL) can mitigate these issues yet typically demands costly real-world interactions or suffers from sim-to-real gaps. We introduce VLA-RFT, a reinforcement fine-tuning framework that leverages a data-driven world model as a controllable simulator. Trained from real interaction data, the simulator predicts future visual observations conditioned on actions, allowing policy rollouts with dense, trajectory-level rewards derived from goal-achieving references. This design delivers an efficient and action-aligned learning signal, drastically lowering sample requirements. **With fewer than 400 fine-tuning steps, VLA-RFT surpasses strong supervised baselines and achieves greater efficiency than simulator-based RL.** Moreover, it exhibits strong robustness under perturbed conditions, sustaining stable task execution. Our results establish world-model-based RFT as a practical post-training paradigm to enhance the generalization and robustness of VLA models. 

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA 12.2+
- PyTorch 2.4+
- UV package manager

### Clone the repository
```bash
# Clone the repository
git clone https://github.com/OpenHelix-Team/VLA-RFT.git
cd VLA-RFT
```
### Installation(If your network is unrestricted)

```bash
# 1) Set up the environment
rm -rf .venv
git submodule update --init --recursive
uv venv --seed -p 3.10
source .venv/bin/activate
python -m pip install -U pip setuptools wheel

# 2) install the shared base FIRST (verl / vLLM side)
# CUDA 12.4 を使う例。環境に合わせて cu121 / cu122 / cu124 は合わせてください。
uv pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0

uv pip install \
  numpy==1.26.4 \
  "fsspec>=2023.1.0,<=2026.2.0" \
  transformers==4.45.0 \
  timm==0.9.10 \
  sentencepiece==0.1.99 \
  accelerate>=0.25.0 \
  peft==0.11.1 \
  protobuf \
  einops \
  rich \
  matplotlib \
  jsonlines \
  huggingface_hub \
  draccus==0.8.0 \
  imageio \
  uvicorn \
  fastapi \
  wandb

# 3) install verl
uv pip install -e train/verl/".[gpu]"
uv pip install -e train/verl/".[vllm]"
uv pip install -r train/verl/requirements.txt

# 4) re-pin shared versions AFTER verl extras
uv pip install \
  numpy==1.26.4 \
  "fsspec>=2023.1.0,<=2026.2.0" \
  transformers==4.45.0 \
  torch==2.4.0 \
  torchvision==0.19.0 \
  torchaudio==2.4.0

# 5) flash-attn
# 既存 wheel を使うなら "torch 2.4.x と CUDA バリアント" を一致させること
# 一致する wheel がない場合は source build にする
uv pip install packaging ninja
uv pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.0.post1/flash_attn-2.6.0.post1+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'

# 6) TensorFlow side needed by openvla-oft / dlimp
uv pip install \
  tensorflow==2.15.0 \
  tensorflow-datasets==4.9.3 \
  tensorflow-graphics==2021.12.3

# 7) install dlimp, but do NOT let it rewrite shared deps
uv pip install git+https://github.com/moojink/dlimp_openvla.git --no-deps

# 8) install openvla-oft code only (important)
uv pip install -e train/verl/vla-adapter/openvla-oft --no-deps

# 9) install LIBERO
# まずは通常。shared deps を壊すようなら --no-deps に切り替える
uv pip install -e third_party/LIBERO

# 10) verify
python - <<'PY'
import torch, torchvision, torchaudio, numpy, transformers
print("torch       =", torch.__version__)
print("torchvision =", torchvision.__version__)
print("torchaudio  =", torchaudio.__version__)
print("numpy       =", numpy.__version__)
print("transformers=", transformers.__version__)
PY

uv pip install json-numpy
uv pip check

```

### Installation(If your network is restricted)
Please refer to the instructions at [third_party/README.md](third_party/README.md).


### Data Preparation
Please refer to the instructions at [data/README.md](data/README.md).

### Basic Usage

#### LIBERO Evaluation Example
```bash
# Run evaluation with LIBERO tasks
cd scripts/libero
bash eval_libero.sh
```
When using LIBERO, you may get an error message like `AttributeError: 'NoneType' object has no attribute 'eglQueryString'`. You can use:

```bash
sudo apt-get update
sudo apt-get install libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglew-dev
```
#### Training Example

```bash
# Run training with LIBERO dataset
cd scripts/libero
bash post_train_rlvr.sh
```

## 📊 Supported Tasks & Benchmarks

### LIBERO Benchmark
- **LIBERO-Spatial**: Spatial reasoning tasks
- **LIBERO-Object**: Object manipulation tasks  
- **LIBERO-Goal**: Goal-conditioned tasks
- **LIBERO-10**: 10-task suite

<div id="top" align="center">
<p align="center">
<img src=image/Figure3.png width=90% />
</p>
</div>

## 📈 Performance

With fewer than 400 fine-tuning steps, VLA-RFT surpasses strong supervised baselines and achieves greater efficiency than simulator-based RL.

<div id="top" align="center">
<p align="center">
<img src=image/Table1.png width=90% />
</p>
</div>

<div id="top" align="center">
<p align="center">
<img src=image/Table2.png width=90% />
</p>
</div>

*Please refer to our paper for detailed benchmark results.*

## 📝 TODO

- [x] Init codebase
- [ ] Release pre-trained and rft VLA(policy) weights
- [ ] Release pre-trained World Model weights
- [ ] Support real-world deployment

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use VLA-RFT in your research, please cite:

```bibtex
@article{wang2025vlaadapter,
  author={Wang, Yihao and Ding, Pengxiang and Li, Lingxiao and Cui, Can and Ge, Zirui and Tong, Xinyang and Song, Wenxuan and Zhao, Han and Zhao, Wei and Hou, Pengxu and Huang, Siteng and Tang, Yifan and Wang, Wenhui and Zhang, Ru and Liu, Jianyi and Wang, Donglin},
  title={VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model},
  journal={arXiv preprint arXiv:2509.09372},
  year={2025}
}
```
```bibtex
@article{li2025vla,
  title={VLA-RFT: Vision-Language-Action Reinforcement Fine-tuning with Verified Rewards in World Simulators},
  author={Li, Hengtao and Ding, Pengxiang and Suo, Runze and Wang, Yihao and Ge, Zirui and Zang, Dongyuan and Yu, Kexian and Sun, Mingyang and Zhang, Hongyin and Wang, Donglin and others},
  journal={arXiv preprint arXiv:2510.00406},
  year={2025}
}
```


## 🙏 Acknowledgments

This work builds upon several excellent open-source projects:
- [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter): Foundation vision-language-action adapter model
- [VERL](https://github.com/volcengine/verl): Volcano Engine Reinforcement Learning framework
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO): Lifelong robot learning benchmark
- [RLVR-world](https://github.com/thuml/RLVR-World): Training world model with verified reward

---

**⭐ Star this repository if you find it helpful!**
