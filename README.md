<div align="center">

# TabDPT: Scaling Tabular Foundation Models on Real Data

[![arxiv](https://img.shields.io/static/v1?label=arXiv&message=2410.18164&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2410.18164)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E)](https://huggingface.co/Layer6/TabDPT)

</div>

**TabDPT** is an open-source foundation model for tabular data based on in-context learning. It is trained on real-world data and can generalize to new tasks **without** additional training or hyperparameter tuning.

This repository provides the full training code to build your own TabDPT model. A lightweight inference interface is available [here](https://github.com/layer6ai-labs/TabDPT-inference), which can support the evaluation of either the existing TabDPT model or any new models that are trained using this repository.


## Usage 

We provide basic usage tips below. The details can be found by stepping through the code.

### Installation

Before running the code, make sure to install the required Python packages:

```
pip install -r requirements.txt
```

You will also need a C compiler such as `gcc` for building some dependencies. On Ubuntu, you can install it with:

```
sudo apt-get update
sudo apt-get install build-essential
```

### Training Example


To train a fresh TabDPT model with default hyperparameters on a single GPU, use the following command:

```
CUDA_VISIBLE_DEVICES=0 python train.py exp_name="TabDPT"
```

#### Multi GPU Training Example

If instead you want to use Multi-GPU, do the following:
```
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:29500 train.py \
  env.gpus="[4,5,6,7]" \
  exp_name="my_multi_gpu_test" 
```

**Notes:**
- Adjust `nproc_per_node` to the number of GPUs.
- If there are communication issues when using several multi gpu training runs on the same node, change the `rdzv_endpoint` port as it can be maxxed out.


## Citation and Acknowledgements

If citing the paper, please use the following BibTeX:

```
@article{ma2024tabdpt,
  title={TabDPT: Scaling Tabular Foundation Models on Real Data},
  author={Ma, Junwei and Thomas, Valentin and Hosseinzadeh, Rasa and Kamkari, Hamidreza and Labach, Alex and Cresswell, Jesse C and Golestan, Keyvan and Yu, Guangwei and Caterini, Anthony L and Volkovs, Maksims},
  journal={arXiv preprint arXiv:2410.18164},
  year={2024}
}
```

Additionally, a huge thank you to [Nafiseh Ghoroghchian](https://github.com/NaGho) for spearheading the effort of refactoring and making this codebase fit for pubilc consumption, and thank you to [Roc Zhang](https://github.com/Zhang-Haipeng) for making the codebase compatible with `safetensors`.
