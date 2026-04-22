# Iteration 2

This directory is the clean workspace for **Iteration 2: Experiment 1**.

Scope:

- input dataset: `data/raw/trolldata.csv`
- task: binary classification
- label mapping: `Normal -> 0`, all other labels -> `1`
- shared split policy: canonical stratified `80/10/10`
- baselines: Logistic Regression, linear SVC, and a PyTorch FFNN

Run order:

```bash
python3 -m src.data.make_dataset --config configs/exp1_binary.yaml
python3 -m src.training.train_binary --config configs/exp1_binary.yaml
```

Artifacts are written to `data/processed/` and `outputs/exp1_binary_baselines/`.
