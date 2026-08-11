# KGL-IDS article implementation

This repository contains only the implementation represented by
[`docs/KG.pdf`](docs/KG.pdf):

```text
raw flow records
    -> KG-inspired relational feature enhancement
    -> train/test split and StandardScaler
    -> SMOTE on the training split only
    -> hybrid GWO-BPSO feature selection
       (Fisher relevance + absolute-correlation redundancy)
    -> lightweight ANN classifier
```

The separate feature-correlation graph prototype, 3GCN/K-GetNID/TKSGF/G-IDCS
comparisons, and ACI-IoT experiments are intentionally not included here.

The corrected WUSTL derivative used in the article is included through Git
LFS. The repository does not redistribute the Edge-IIoT dataset. Download
instructions, source links, expected filenames, and checksums are documented
in [`data/README.md`](data/README.md).

## Datasets

The code expects:

- `data/raw/wustl/wustl_corrected.csv`
- `data/raw/edgeiiot/DNN-EdgeIIoT-dataset.csv`

WUSTL-IIoT uses the grouped classes `Backdoor`, `CommInj`, `DoS`, `Reconn`,
and `normal`. Edge-IIoT uses `Backdoor_MITM`, `Brute_Force`, `DDoS_DoS`,
`Malware`, `Normal`, `Recon`, and `Web_Attack`.

## Setup

```bash
cd KGL_IDS_article_implementation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare and validate data

Follow [`data/README.md`](data/README.md), then validate the inputs:

```bash
python scripts/00_validate_data.py --dataset wustl
python scripts/00_validate_data.py --dataset edgeiiot
```

When cloning the repository, retrieve the corrected WUSTL LFS object before
validation:

```bash
git lfs pull
```

## Run the article pipeline

Run each dataset independently:

```bash
python scripts/01_preprocess.py --dataset wustl
python scripts/02_smote.py --dataset wustl
python scripts/03_select_features.py --dataset wustl
python scripts/04_train_evaluate.py --dataset wustl --variant kgl_ids
```

For Edge-IIoT:

```bash
python scripts/01_preprocess.py --dataset edgeiiot
python scripts/02_smote.py --dataset edgeiiot
python scripts/03_select_features.py --dataset edgeiiot
python scripts/04_train_evaluate.py --dataset edgeiiot --variant kgl_ids
```

The training script also supports the ablation variants:

```bash
python scripts/04_train_evaluate.py --dataset wustl --variant baseline
python scripts/04_train_evaluate.py --dataset wustl --variant kg
python scripts/04_train_evaluate.py --dataset wustl --variant kg_smote
```

## Outputs

Outputs are written under `data/processed/<dataset>/`:

```text
data/processed/<dataset>/
├── baseline/
├── kg/
├── smote/
├── selected/
└── models/
```

The selected-feature stage records the selected names, Fisher scores, search
parameters, and the final train/test Parquet files. The training stage records
the ANN checkpoint, history, metrics, classification report, and confusion
matrix.

The paper-level reference values are recorded in
[`results/REFERENCE_RESULTS.md`](results/REFERENCE_RESULTS.md). They are
provided as a comparison target; the training script computes fresh metrics.

## Article-faithful details

- The included WUSTL derivative corrects `IdleTime`: for each flow occurrence,
  it represents the time difference between the current flow's start and the
  end of the previous occurrence of that flow, rather than storing that prior
  end time itself.
- Numeric features, including the corrected `IdleTime`, are standardized with
  a `StandardScaler` fitted on the training split and then applied to the test
  split.
- KG features use source/destination entities, ports/protocol-like fields, and
  fixed 5-second windows.
- IP/identity and timestamp fields are retained while KG features are built and
  removed afterward.
- KG aggregation is performed before the split, matching the current article
  implementation.
- StandardScaler is fitted on the training split and applied to the test split.
- SMOTE uses `random_state=42`, `k_neighbors=5`, and training data only.
- HGWO-BPSO uses population 30, 50 iterations, `w` from 0.9 to 0.4,
  `c1=c2=1.8`, position clipping `[-6, 6]`, binary threshold 0.5, and
  `lambda=0.995`.
- The ANN uses Dense layers `256 -> 128 -> 64`, batch normalization, ReLU,
  dropout 0.3, Adam at `1e-3`, validation split 0.1, early stopping, and
  learning-rate reduction.
