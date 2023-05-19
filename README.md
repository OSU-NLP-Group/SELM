# SELM

Code and data for [SELM](https://samuelstevens.me/research/encryption) research project.

![teaser-gif-cropped](https://github.com/OSU-NLP-Group/SELM/assets/26638161/b7484c1f-84da-45a9-ba69-0c921c5d87cf)

## Table of Contents

1. Introduction
2. Installation
3. Encrypt Something
4. Decrypt Something
5. Experiments
6. Cryptanalysis

## Installation

Install torch (CUDA):

```
pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Do this first before installing `requirements.txt` because that will install a CPU-only torch.

Install packages:

```sh
pip install -r requirements.txt
```

Install the intrinsic package, used for efficient intrinsic dimension operations:

```sh
cd intrinsic
python setup.py develop
cd ..
```

Initialize `relics/` (the experiment directory):

```sh
relic init
```

## Encrypt Something

Get a key:

```sh
python -c 'import secrets; print(secrets.randbits(32))'
```

Encrypt with your key:

```sh
python encrypt.py --key KEY --int-dim 10000 data/examples/advil.txt
```

## Decrypt Something

Use the key to decrypt:

```
python decrypt.py --key KEY advil.bin
```

## Experiments

To run a new experiment, define a new `.toml` file in `experiments/` with whatever configuration options you want. `src/config.py` shows all the different options that can be changed.

`.toml` files can contain lists for parameters; when they do, an experiment for each value in the list is generated. For example, `experiments/gpt2/wikipedia/0-4-concat.toml` has two lists: one for `learning_rate` and `intrinsic_dimension`. This means there are actually 20 experiments in here: 2 learning rates * 10 intrinsic dimensions.

To run the experiments:

```sh
python -m src.experiments.run experiments/templates/paper/what-can-we-encrypt-v4.toml
```

If you are running out of GPU memory, you can use model parallelism to split the Fastfood transform and the GPT2 model onto separate GPUs:

```
CUDA_VISIBLE_DEVICES=0,2 MODEL_PARALLELISM=1 python -m src.experiments.run experiments/gpt2/examples/medium.toml
```

You can pass entire directories or just individual `.toml` files to `src.experiments`. Results will be saved to `relics/`.

**If you stop an experiment and run it again, any trials that are finished in `relics/` will not be run again.**

## Cryptanalysis

Unzip the provided data:

```sh
unzip relics.zip
```

Play the security game on the original algorithm with an SVM:

```sh
python -m src.paper.security svm original feature-fn 500 --ratio 0.8 --quiet
```

Play the security game on the distribution-regularized variant with an SVM:

```sh
python -m src.paper.security svm distribution-reg feature-fn 500 --ratio 0.8 --quiet
```

Try to implement stronger attacks!
Look in `src/attacking/` for the model files and add your own.
