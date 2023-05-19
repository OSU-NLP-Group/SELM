# How to Reproduce

This doc is about how to reproduce the charts, graphs, and tables in the paper.

## Data

Generate the texts used in the semantic security games:

```sh
python -m src.data random-sentences --output data/random-sentences --seed 42 --input data/openwebtext/ data/twitter/ data/reddit
python -m src.data random-words --output data/random-words/ --seed 42
python -m src.data random-letters --output data/random-letters/ --seed 42
python -m src.data random-bytes --output data/random-bytes/ --seed 42
```

## Experiments


Generate the template files

```
mkdir -p experiments/generated/paper

python -m src.experiments.generate experiments/templates/paper/original-algorithm-v3.toml experiments/generated/paper
python -m src.experiments.generate experiments/templates/paper/l2-norm-v3.toml experiments/generated/paper
python -m src.experiments.generate experiments/templates/paper/distribution-regularization-v4.toml experiments/generated/paper
```

Encrypt the plaintexts:

```
python -m src.experiments.run experiments/templates/paper/*
```

## Tables

Original algorithm:

```sh
python -m src.paper.security knn original feature-fn 500 --ratio 0.8 --quiet
python -m src.paper.security knn original cipher 500 --ratio 0.8 --quiet
python -m src.paper.security lda original feature-fn 500 --ratio 0.8 --quiet
python -m src.paper.security lda original cipher 500 --ratio 0.8 --quiet
python -m src.paper.security svm original feature-fn 500 --ratio 0.8 --quiet
python -m src.paper.security svm original cipher 500 --ratio 0.8 --quiet
python -m src.paper.security gradboost original feature-fn 500 --ratio 0.8 --quiet
python -m src.paper.security gradboost original cipher 500 --ratio 0.8 --quiet
python -m src.paper.security ffnn original feature-fn 500 --ratio 0.8 --quiet
python -m src.paper.security ffnn original cipher 500 --ratio 0.8 --quiet
```

L2-norm regularization:

```sh
python -m src.paper.security knn l2-norm-reg feature-fn 500 --ratio 0.8 --quiet
python -m src.paper.security knn l2-norm-reg cipher 500 --ratio 0.8 --quiet
python -m src.paper.security lda l2-norm-reg feature-fn 500 --ratio 0.8 --quiet
python -m src.paper.security lda l2-norm-reg cipher 500 --ratio 0.8 --quiet
python -m src.paper.security svm l2-norm-reg feature-fn 500 --ratio 0.8 --quiet
python -m src.paper.security svm l2-norm-reg cipher 500 --ratio 0.8 --quiet
python -m src.paper.security gradboost l2-norm-reg feature-fn 500 --ratio 0.8 --quiet
python -m src.paper.security gradboost l2-norm-reg cipher 500 --ratio 0.8 --quiet
python -m src.paper.security ffnn l2-norm-reg feature-fn 500 --ratio 0.8 --quiet
python -m src.paper.security ffnn l2-norm-reg cipher 500 --ratio 0.8 --quiet
```

Distribution regularization:

```sh
python -m src.paper.security knn distribution-reg feature-fn 500 --ratio 0.8 --quiet
python -m src.paper.security knn distribution-reg cipher 500 --ratio 0.8 --quiet
python -m src.paper.security lda distribution-reg feature-fn 500 --ratio 0.8 --quiet
python -m src.paper.security lda distribution-reg cipher 500 --ratio 0.8 --quiet
python -m src.paper.security svm distribution-reg feature-fn 500 --ratio 0.8 --quiet
python -m src.paper.security svm distribution-reg cipher 500 --ratio 0.8 --quiet
python -m src.paper.security gradboost distribution-reg feature-fn 500 --ratio 0.8 --quiet
python -m src.paper.security gradboost distribution-reg cipher 500 --ratio 0.8 --quiet
python -m src.paper.security ffnn distribution-reg feature-fn 500 --ratio 0.8 --quiet
python -m src.paper.security ffnn distribution-reg cipher 500 --ratio 0.8 --quiet
```

## Figures

```sh
# What can we encrypt
python -m src.paper.what_can_we_encrypt domain docs/paper/src/figures/domain.pdf
python -m src.paper.what_can_we_encrypt length docs/paper/src/figures/length.pdf
python -m src.paper.what_can_we_encrypt size docs/paper/src/figures/size.pdf

# Ciphertext embeddings
python -m src.paper.embeddings original 500 docs/paper/src/figures/original-tsne.pdf
python -m src.paper.embeddings l2-norm-reg 500 docs/paper/src/figures/l2-norm-reg-tsne.pdf
python -m src.paper.embeddings distribution-reg 500 docs/paper/src/figures/distribution-reg-tsne.pdf

# Measure feature importance
python -m src.paper.feature_importance 500 docs/paper/src/figures/mi.pdf
```
