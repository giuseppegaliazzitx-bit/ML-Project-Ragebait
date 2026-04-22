# Rage-Bait Detector

This repository has been reset around **Iteration 2**, the fresh supervised ML track built from `trolldata.csv`.

The active project now lives in [`iteration2/`](./iteration2). It is a clean, standalone workspace for:

- canonical dataset splitting
- binary baseline training
- validation-time evaluation and confusion matrices
- structured experiment outputs

Iteration 1 has been preserved as archived legacy material and renamed so it is no longer the active surface area:

- `README_iteration1_legacy.md`
- `legacy_iteration1_configs/`
- `legacy_iteration1_manual_eval_app/`
- `legacy_iteration1_outputs/`
- `legacy_iteration1_pyproject.toml`
- `legacy_iteration1_ragebait_detector/`
- `legacy_iteration1_ragebait_detector.egg-info/`
- `legacy_iteration1_scripts/`
- `legacy_iteration1_tests/`

## Active Layout

```text
iteration2/
  configs/
    exp1_binary.yaml
  data/
    raw/
      trolldata.csv
    processed/
      binary_train.csv
      binary_val.csv
      binary_test.csv
      binary_split_manifest.json
  outputs/
    exp1_binary_baselines/
      summary.json
  src/
    data/
      make_dataset.py
      preprocessing.py
    models/
      baselines.py
    training/
      train_binary.py
    evaluation/
      evaluate.py
```

## Iteration 2 Quick Start

```bash
cd iteration2
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m src.data.make_dataset --config configs/exp1_binary.yaml
python3 -m src.training.train_binary --config configs/exp1_binary.yaml
```

The old Iteration 1 documentation is still available in `README_iteration1_legacy.md`, but the new pipeline does not import or depend on that code.
