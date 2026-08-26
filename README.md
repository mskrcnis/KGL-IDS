# KGL-IDS: Knowledge Graph–Inspired Intrusion Detection for IIoT

This repository contains the code and reproducibility materials for the KGL-IDS article. The implementation evaluates a lightweight intrusion-detection pipeline on the WUSTL-IIoT-2021 and DNN-Edge-IIoT datasets.

The method is a tabular, flow-level IDS. It uses source/destination and protocol relationships to construct knowledge-graph-inspired relational features; it does not build a graph neural network and it does not require PCAP files.

## Project overview

The complete pipeline is:

~~~text
flow-level CSV data
    -> cleaning, categorical encoding, and target construction
    -> baseline features and KG-inspired relational features
    -> stratified 80/20 train/test split
    -> StandardScaler fitted on the training split
    -> SMOTE applied to the training split only
    -> hybrid GWO-BPSO feature selection
       using Fisher relevance and correlation redundancy
    -> fully connected ANN classifier
    -> accuracy, precision, recall, F1, report, and confusion matrix
~~~

The paper and the paper-level reference values are included in:

- `docs/KG.pdf` — article manuscript.
- `results/REFERENCE_RESULTS.md` — reported comparison values and approximate stored metrics.

## Repository structure

~~~text
KGL-IDS/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   ├── SHA256SUMS.txt
│   └── raw/
│       ├── wustl/wustl_corrected.csv
│       └── edgeiiot/DNN-EdgeIIoT-dataset.csv   # downloaded by the user
├── docs/KG.pdf
├── kgl_ids/
│   ├── config.py          # dataset paths and column configuration
│   ├── preprocessing.py   # cleaning, targets, scaling, and KG features
│   ├── balancing.py       # training-only SMOTE
│   ├── selection.py       # hybrid GWO-BPSO feature selection
│   └── model.py           # ANN training and evaluation
├── scripts/
│   ├── 00_validate_data.py
│   ├── 01_preprocess.py
│   ├── 02_smote.py
│   ├── 03_select_features.py
│   └── 04_train_evaluate.py
└── results/REFERENCE_RESULTS.md
~~~

Generated Parquet files, models, and metrics are written to `data/processed/` and are not required to be prepared manually.

## Dataset information

### WUSTL-IIoT-2021

- Original dataset: WUSTL-IIoT-2021 Dataset for IIoT Cybersecurity Research.
- Original source: https://www.cse.wustl.edu/~jain/iiot2/index.html
- Alternate current WUSTL page: https://research.engineering.wustl.edu/~jain/iiot2/index.html
- Repository file: `data/raw/wustl/wustl_corrected.csv`.
- Included in this repository through Git LFS.
- Multiclass label: `Traffic`.
- Binary label retained by the preprocessing artifacts: `Target`.

The repository file is the corrected derivative used in the study, not an unmodified replacement downloaded from the original source. In the original release, `IdleTime` was incorrectly recorded as the end time of the previous occurrence of the same flow. The corrected derivative makes it the time difference between the current occurrence's `StartTime` and the previous occurrence's `LastTime`. If the first occurrence has no previous matching flow, its idle time is retained as zero.

The included file has SHA-256 checksum:

~~~text
00963840a33aec2063eaf1b248617f64ee3fa74745ff48526883c8062a991b74  data/raw/wustl/wustl_corrected.csv
~~~

Do not substitute the original uncorrected file if exact reproduction of the reported WUSTL-IIoT results is required. The original source page also recommends removing identity and timestamp fields because they can expose attack identity. This implementation retains them temporarily for KG feature construction and removes them before feature modeling; the baseline removes them before modeling.

### DNN-Edge-IIoT

- Dataset: Edge-IIoTset Cyber Security Dataset of IoT and IIoT.
- Public source: https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot
- Required local filename: `DNN-EdgeIIoT-dataset.csv`.
- Required local path: `data/raw/edgeiiot/DNN-EdgeIIoT-dataset.csv`.
- Not redistributed in this repository; download it from the public Kaggle source.
- Multiclass source label: `Attack_type`.
- Binary source label: `Attack_label`.

After downloading the Kaggle file, preserve its filename and place it at the path above. The validator checks the expected schema before the pipeline is run.

Neither dataset is supplied as PCAP. The implementation consumes the published flow-level CSV files and does not perform packet capture or packet-to-flow extraction.

More detailed dataset placement and validation notes are in [`data/README.md`](data/README.md).

## Requirements

- Python 3.10 or newer; Python 3.10–3.12 is recommended.
- Git LFS, required to retrieve the included corrected WUSTL CSV.
- A machine with enough disk space and memory for the full CSV-to-Parquet transformation. The Edge-IIoT CSV is large.
- Python packages listed in `requirements.txt`: NumPy, pandas, PyArrow, scikit-learn, imbalanced-learn, joblib, and TensorFlow.

## Installation

Clone the repository and retrieve the WUSTL file:

~~~bash
git clone https://github.com/mskrcnis/KGL-IDS.git
cd KGL-IDS
git lfs install
git lfs pull
~~~

Create an isolated Python environment and install the dependencies:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

## Data preparation and validation

1. Confirm that the corrected WUSTL file is present at `data/raw/wustl/wustl_corrected.csv` after `git lfs pull`.
2. Download `DNN-EdgeIIoT-dataset.csv` from Kaggle.
3. Place it at `data/raw/edgeiiot/DNN-EdgeIIoT-dataset.csv`.
4. Validate each input before preprocessing:

~~~bash
python scripts/00_validate_data.py --dataset wustl
python scripts/00_validate_data.py --dataset edgeiiot
~~~

To verify the included WUSTL file independently:

~~~bash
sha256sum -c data/SHA256SUMS.txt
~~~

## Running the complete pipeline

Run the following stages in order for each dataset. The stages are intentionally separate so that intermediate data, selected features, and metrics can be inspected.

### WUSTL-IIoT

~~~bash
python scripts/01_preprocess.py --dataset wustl
python scripts/02_smote.py --dataset wustl
python scripts/03_select_features.py --dataset wustl
python scripts/04_train_evaluate.py --dataset wustl --variant kgl_ids
~~~

### DNN-Edge-IIoT

~~~bash
python scripts/01_preprocess.py --dataset edgeiiot
python scripts/02_smote.py --dataset edgeiiot
python scripts/03_select_features.py --dataset edgeiiot
python scripts/04_train_evaluate.py --dataset edgeiiot --variant kgl_ids
~~~

The final `kgl_ids` variant uses the selected features and SMOTE-balanced training data. The training script creates the ANN checkpoint and evaluation artifacts in the corresponding dataset model directory.

## Ablation variants

`scripts/04_train_evaluate.py` also supports the following variants:

~~~bash
# Raw baseline features, without KG enhancement or SMOTE
python scripts/04_train_evaluate.py --dataset wustl --variant baseline

# KG-enhanced features, without SMOTE
python scripts/04_train_evaluate.py --dataset wustl --variant kg

# KG-enhanced features with SMOTE, before feature selection
python scripts/04_train_evaluate.py --dataset wustl --variant kg_smote
~~~

Replace `wustl` with `edgeiiot` for the second dataset. The preprocessing, SMOTE, and selection stages must be run first.

## Methodology implemented

### Preprocessing

- WUSTL rows with zero `Dur` are removed.
- Infinite values are replaced, numeric missing values are filled with zero, and remaining categorical missing values are filled with the mode.
- Categorical protocol/text fields are label encoded.
- WUSTL uses the five traffic groups present in `Traffic`.
- Edge-IIoT raw attack labels are mapped to the article's higher-level groups: `Backdoor_MITM`, `Brute_Force`, `DDoS_DoS`, `Malware`, `Normal`, `Recon`, and `Web_Attack`.
- A stratified 80/20 split is generated with `random_state=42`.
- `StandardScaler` is fitted using training features only and then applied to the test features.

### KG-inspired relational features

Relational features are derived from source and destination entities, protocol-like fields, ports, and fixed five-second time windows. The implementation includes flow counts, unique-peer and unique-port/protocol counts, pair counts, time-window activity counts, aggregate packet/byte/duration measures, and diversity/focus ratios. These are tabular relational features; no graph neural network is used.

KG aggregation is performed before the train/test split to match the article implementation. Identity and timestamp columns are removed after KG features are constructed.

### Balancing and feature selection

- SMOTE is applied to the KG training split only, using `random_state=42` and `k_neighbors=5`. The test split is never oversampled.
- The selector uses a population of 30 for 50 iterations.
- Fisher scores measure class relevance.
- Absolute feature correlations form the redundancy penalty.
- GWO/PSO parameters are `w=0.9` to `0.4`, `c1=c2=1.8`, position clipping `[-6, 6]`, binary threshold `0.5`, and penalty `lambda=0.995`.
- The selector writes the selected names, Fisher scores, correlation matrix, search parameters, and selected train/test files.

### ANN classifier

The classifier is a fully connected ANN with Dense layers `256 -> 128 -> 64`, batch normalization, ReLU activations, dropout `0.3`, and a final softmax layer. It uses Adam with learning rate `1e-3`, batch size `4096`, up to 50 epochs, validation split `0.1`, early stopping, and learning-rate reduction. The model seed is `123`.

## Outputs

For each dataset, outputs are written under:

~~~text
data/processed/<dataset>/
├── baseline/
│   ├── cleaned.parquet
│   ├── train_group.parquet
│   └── test_group.parquet
├── kg/
│   ├── cleaned.parquet
│   ├── train_group.parquet
│   └── test_group.parquet
├── smote/
│   └── train_group_smote.parquet
├── selected/
│   ├── selected_features.json
│   ├── fisher_scores.csv
│   ├── correlation_matrix.npy
│   ├── train_df_group_selected.parquet
│   └── test_df_group_selected.parquet
└── models/<variant>/
    ├── best_ann.keras
    ├── training_history.csv
    ├── metrics.json
    ├── classification_report.txt
    ├── classification_report.csv
    ├── confusion_matrix.csv
    └── test_predictions.csv
~~~

Intermediate files also include fitted scalers, feature encoders, class mappings, and unscaled split files.

## Reference results

The reported article-level comparison values are:

| Dataset | Selected features | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| WUSTL-IIoT | 23 | 0.9999 | 0.9962 |
| Edge-IIoT | 27 | 0.9992 | 0.9919 |

Approximate stored experiment values are WUSTL-IIoT accuracy `0.999987` and macro-F1 `0.996158`, and Edge-IIoT accuracy `0.999205` and macro-F1 `0.991902`. Fresh runs can differ slightly across TensorFlow, CUDA, hardware, and dependency versions. See [`results/REFERENCE_RESULTS.md`](results/REFERENCE_RESULTS.md).

## Citations and data availability

When using this repository, cite the associated article represented by [`docs/KG.pdf`](docs/KG.pdf), the original WUSTL-IIoT-2021 dataset, and the original Edge-IIoTset dataset.

Dataset source records:

- WUSTL-IIoT-2021: https://www.cse.wustl.edu/~jain/iiot2/index.html
- Edge-IIoTset: https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot
- Code repository: https://github.com/mskrcnis/KGL-IDS

The corrected WUSTL derivative is included in this repository through Git LFS. The Edge-IIoT file is obtained by the user from Kaggle and is intentionally not redistributed here. Dataset access, use, and redistribution remain subject to the terms of the respective source providers.

## License and contributions

This repository does not currently include a separate software license. The dataset files remain subject to the terms of their original providers; inclusion of the corrected WUSTL derivative does not change those terms. Please contact the authors before redistributing repository data or using the code outside the study's reproducibility purpose.

Bug reports and reproducibility improvements are welcome through GitHub issues and pull requests. Contributions should include the affected command, environment details, and validation output. Do not commit downloaded Edge-IIoT data, generated Parquet/model artifacts, credentials, or unrelated files.

## Contact

For questions about the article implementation or the corrected WUSTL derivative, please open an issue in this repository or contact the corresponding author listed in the article.