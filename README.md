# Explainable and Robust Deep Learning for Anomaly Detection in Industrial Control Systems: A WDNN-GRU Comparison

This repository contains a scoped comparative investigation of prediction-error anomaly detection on the Secure Water Treatment (SWaT) dataset. It includes a **DAICS-inspired wide-and-deep neural network (WDNN) baseline** and a separately trained **GRU-based temporal alternative**. The GRU is an independent model; it is not integrated into DAICS and does not replace the WDNN implementation.

This work is not a complete reproduction of the original DAICS paper. The implemented approximations and data limitations are listed below and in [`report/limitations_and_verification.md`](report/limitations_and_verification.md).

## Project objective

Industrial control systems (ICS) combine sensors, actuators, controllers, and physical processes. Unusual sensor/actuator behaviour can indicate faults or cyberattacks. This project compares two predictors trained on normal SWaT observations, calibrates model-specific error thresholds, evaluates anomaly detection on separate normal and attack sources, attributes section prediction residuals to sensors, and tests the predictors under controlled Gaussian sensor noise.

## Models and detection

### WDNN baseline

The DAICS-inspired WDNN uses an input window of 60 observations over 44 process variables and predicts four future values for each retained sensor. Its three output sections contain 5, 7, and 12 sensors. The WDNN input has shape `(batch, 44, 60)`. Its wide branch flattens the input and applies a learned dense projection to four values; this approximates, but does not explicitly construct, the paper's feature crosses. The deep branch uses `DL2=180`, Conv1D with 64 kernels of size 2 and max-pooling by 2, then Conv1D with 128 kernels of size 2 and max-pooling by 2, followed by `DL3=4`. The four wide and four deep values are concatenated and passed through `DL4=80`. Each section head uses `DL5=round(2.25*m)`, `DL6=round(1.5*m)`, `DL7=m`, followed by a linear `m x 4` output, where `m` is 5, 7, or 12. Hidden layers use LeakyReLU with alpha 0.01.

The WDNN configuration follows the project configuration: SGD, learning rate 0.01, batch size 32, and 100 epochs. Section mean-squared-error losses are summed. Evaluation uses the saved epoch-100 checkpoint. Epoch 90 had the best logged validation loss, but its weights were not saved.

### Independent GRU alternative

The separate GRU receives the same 44-variable, 60-step input convention, processes the sequence with one GRU layer (hidden size 64), and has three independent output heads for 5, 7, and 12 sensors, each forecasting four values. The recorded training setup is Adam with learning rate 0.001, batch size 32, maximum 15 epochs, early-stopping patience 3, gradient clipping 1.0, and seed 20261101. The selected checkpoint is epoch 12. The models are evaluated independently with their own TTNN calibration and thresholds.

### Error thresholds and actuator states

For each model, prediction errors are calculated by section. The validation-derived base threshold is `T_base = mean(validation_MSE) + std(validation_MSE)`. Historical MSE windows have length 60 and use a median filter of size 59 (the implementation uses per-window filtering with reflected boundaries). A separate TTNN is calibrated per section and contributes to a historical, section-specific threshold of the form `T_g = T_base + max(TTNN(historical_MSE_windows_g))`. The TTNN target uses next-MSE regression as an explicitly documented approximation because the paper does not define its supervised target/loss unambiguously. Test-time online TTNN updates are not performed. Sensor threshold alarms are combined with actuator-state database alarms and the configured `Wanom = 30` consecutive-sample rule. The normal actuator database contains 70 combinations in this dataset, versus 146 reported in the paper.

## Dataset and preprocessing

SWaT is a water-treatment ICS dataset with normal-operation and controlled-attack recordings. Obtain it only through an authorized dataset source and comply with its terms; the data are intentionally excluded from this repository.

The local preprocessing removed 495,000 exact duplicate normal rows, excluded seven unavailable process variables (`MV101`, `AIT201`, `MV201`, `P201`, `P202`, `P204`, and `MV303`), checked remaining values, and sorted records by timestamp. The resulting files contain 44 process variables (24 sensors and 20 actuators), timestamp, and label. Sensor scaling is fitted on the normal training partition only; actuator states remain unscaled. Normal and attack test sources are windowed independently. The merged CSV is a derived combination and is not an independent test source.

| Source | Rows after preprocessing | Role |
|---|---:|---|
| Normal | 892,098 | Normal-only training, validation, and remaining normal test period |
| Attack | 54,621 | Separate attack test stream; labels retained |
| Merged | 946,719 | Derived combined file; not used as an independent test stream |

The approved chronological Option A split uses the first seven days of normal observations: 478,365 training rows and 119,591 benign validation rows. The remaining normal test source has 294,142 rows; the independent attack source has 54,621 rows. The validation timestamps overlap the calendar period covered by the separately sourced attack recording, but attack-source rows are not used for training or calibration. Source-local test window counts are 294,029 normal and 54,508 attack windows. Window parameters are `Win=60`, `H=50`, `Wout=4`, and `Wanom=30`.

Sensor sections are stage-based approximations (5/7/12 sensors), not verified PLC groupings. The paper used 51 process variables; seven unavailable variables were excluded here.

## Explainability

Residual attribution computes squared error per sensor and averages across the four forecast steps. Within each affected sensor section, the sensor residual is expressed as a percentage of that section's total residual and ranked. The output identifies sensors contributing most to the model's prediction error for a detected section episode. It is not a causal explanation and does not identify attack responsibility. Test labels are not used to calculate attribution.

## Noise robustness experiment

The evaluation adds independent zero-mean Gaussian noise to normalized sensor readings only, at standard deviations of 0, 0.01, 0.05, and 0.10 times the corresponding training sensor standard deviation (clean, low, medium, high). Actuator states and labels are unchanged; values are not clipped. Saved fixed-seed noise realizations are shared between WDNN and GRU. The models use their own clean-validation-derived threshold calibrations; clean-reference threshold trajectories are frozen across noise conditions rather than adapted from noisy history.

## Results

The following are pooled point-based results read from the saved clean evaluation and robustness CSV files. Values are rounded to four decimals; confusion counts are included for the clean condition.

### Clean test comparison

| Model | Precision | Recall | F1 | FPR | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WDNN baseline | 0.5153 | 0.8818 | 0.6505 | 0.1538 | 48,067 | 45,214 | 248,815 | 6,441 |
| Independent GRU | 0.6675 | 0.8261 | 0.7383 | 0.0763 | 45,028 | 22,434 | 271,595 | 9,480 |

The GRU has higher pooled F1 and lower FPR in the saved clean evaluation; WDNN has higher recall. Normal-only test data contain no attack-positive labels, so their F1 is not informative; FPR is the relevant normal-source false-alarm measure.

### Robustness comparison

| Noise | WDNN precision | WDNN recall | WDNN F1 | WDNN FPR | GRU precision | GRU recall | GRU F1 | GRU FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean | 0.5153 | 0.8818 | 0.6505 | 0.1538 | 0.6675 | 0.8261 | 0.7383 | 0.0763 |
| Low (0.01) | 0.5156 | 0.8818 | 0.6507 | 0.1536 | 0.6670 | 0.8260 | 0.7380 | 0.0764 |
| Medium (0.05) | 0.5178 | 0.8813 | 0.6523 | 0.1521 | 0.6796 | 0.8251 | 0.7453 | 0.0721 |
| High (0.10) | 0.5224 | 0.8804 | 0.6557 | 0.1492 | 0.6975 | 0.8238 | 0.7554 | 0.0662 |

Across the tested conditions, GRU has higher pooled F1 and lower FPR, while WDNN has higher attack recall. Both models show slightly lower attack recall as noise increases. The increase in pooled F1 is associated mainly with fewer false positives on normal windows; it should not be interpreted as noise improving attack detection. Detailed per-source metrics and confusion counts are in [`results/curated/`](results/curated/).

The logged detection delay is zero seconds because the available attack test labels form one contiguous labeled interval and the detector flags the first labeled point. This is not a meaningful per-attack latency estimate for the paper's 36 individual attacks.

## Limitations and deviations

- This is not a complete reproduction of the original DAICS paper.
- The implementation uses 44 available process variables instead of the paper's 51; seven unavailable variables were excluded.
- The 5/7/12 sensor sections are stage-based approximations, not a confirmed PLC mapping.
- The WDNN wide path approximates the paper's incompletely specified feature-cross operation.
- TTNN next-MSE regression is an implementation approximation; the paper does not clearly specify the supervised target/loss.
- The paper's online TTNN update during test operation is not reproduced; test thresholds use historical errors and fixed clean-validation calibration.
- Few-time-steps WDNN adaptation is not implemented.
- The actuator database contains 70 combinations rather than the paper's reported 146.
- WDNN testing uses epoch 100 because epoch-90 weights (best logged validation loss) were not saved. GRU training ran for 15 epochs with the best saved model from epoch 12; epoch counts are not directly comparable across architectures.
- Attack labels do not preserve individual attack intervals, limiting attack-level detection-delay interpretation.
- Residual attribution is prediction-error contribution, not causal responsibility.
- Robustness thresholds are frozen clean-reference trajectories; noisy-history threshold adaptation is not evaluated.
- CUDA execution is not guaranteed bitwise deterministic across platforms.

See [`report/limitations_and_verification.md`](report/limitations_and_verification.md) for details requiring manual confirmation. Future work may investigate closer WDNN feature-cross reconstruction, a fully specified TTNN update objective, few-time-steps adaptation, per-attack labels, other explainability methods, and additional ICS datasets. These extensions are not implemented here.

## Installation

The project was run with Python 3.12.10 and PyTorch 2.5.1+cu121. Create a Windows environment from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch separately to select the appropriate CPU/CUDA wheel. The recorded CUDA environment used PyTorch 2.5.1 with CUDA 12.1:

```powershell
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

For CPU-only use, install the matching CPU PyTorch build using the official PyTorch installation selector, then install `requirements.txt`. Check the runtime with:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Data setup

Place authorized source files at `data/SWaT/normal.csv`, `data/SWaT/attack.csv`, and `data/SWaT/merged.csv`. The preprocessing script reads these three files and writes cleaned versions plus a JSON report to `data/processed/`; it overwrites outputs at those paths. Follow [`data/README.md`](data/README.md). Do not commit the source or processed CSV files.

## Execution

Run commands from the repository root in the activated environment. They read and write the fixed relative paths shown below. Back up or use a clean clone before rerunning commands that overwrite trained models/results.

```powershell
# Preprocess locally obtained SWaT CSVs (writes/overwrites data/processed/*)
python preprocess_swat.py

# Train WDNN baseline (100 epochs; checkpoint and log paths are under models/ and logs/)
python -m daics.training

# Calibrate WDNN-specific TTNN/thresholds from benign validation errors
python calibrate_ttnn.py

# Evaluate saved WDNN artifacts on source-separated clean test data
python evaluate_baseline.py

# Train the independent GRU alternative (up to 15 epochs; best validation checkpoint)
python -m daics.training_gru

# Calibrate a fresh GRU-specific TTNN on benign validation errors and run clean evaluation
python evaluate_gru.py

# Explain saved detections (after the corresponding clean evaluation)
python explain_baseline.py
python explain_gru.py

# Robustness evaluation; run WDNN first so saved noise realizations are reused by GRU
python evaluate_robustness.py
python evaluate_gru_robustness.py

# Unit tests (temporary test data only)
python -m unittest discover -s tests
```

`python -m daics.training` resumes from `models/wdnn_checkpoint.pt` when that file exists. To train from scratch, use a clean working copy without that checkpoint. The GRU training path writes its best checkpoint to `checkpoints/gru/best_checkpoint.pt`; preserve or stage existing artifacts before retraining. `evaluate_gru.py` performs GRU TTNN calibration as part of its evaluation run. Robustness scripts write result files and may create missing noise realization arrays; they do not train models.

## Using the published checkpoints for inference

The verified checkpoints are kept as a provenance-labeled bundle under `results/curated/checkpoints/`; the working paths under `models/` and `checkpoints/` are ignored to avoid uploading duplicate or smoke-test artifacts. Evaluation scripts expect the original local paths. On a clean checkout with the data already prepared, stage the WDNN assets before `evaluate_baseline.py` (PowerShell):

```powershell
New-Item -ItemType Directory -Force models, data/processed, checkpoints/gru | Out-Null
Copy-Item results/curated/checkpoints/wdnn_checkpoint_epoch100.pt models/wdnn_checkpoint.pt -NoClobber
Copy-Item results/curated/checkpoints/wdnn_ttnn_checkpoint.pt models/ttnn_checkpoint.pt -NoClobber
Copy-Item results/curated/checkpoints/wdnn_thresholds.joblib models/thresholds.joblib -NoClobber
Copy-Item results/curated/checkpoints/wdnn_calibration_metadata.json models/thresholds.json -NoClobber
Copy-Item results/curated/checkpoints/training_fitted_sensor_scaler.joblib models/norm_params.joblib -NoClobber
Copy-Item results/curated/checkpoints/wdnn_actuator_database.joblib data/processed/actuator_db.joblib -NoClobber
```

For GRU clean evaluation, also stage `gru_best_checkpoint_epoch12.pt` as `checkpoints/gru/best_checkpoint.pt`; `evaluate_gru.py` recalibrates its own TTNN from benign validation errors. For GRU robustness evaluation, stage the GRU TTNN checkpoint and thresholds as `checkpoints/gru/ttnn_checkpoint.pt` and `checkpoints/gru/thresholds.joblib` as well. The GRU calibration metadata is in the curated bundle. `-NoClobber` prevents accidental replacement of local artifacts. The provenance manifest lists the original source paths, selection epochs, and known limitations.

## Curated publication artifacts

[`results/curated/`](results/curated/) contains compact evaluation summaries, selected plots, the preprocessing report, and a provenance manifest for the exact checkpoints/calibration files used by the reported evaluations. It excludes the SWaT records, point-level prediction dumps, and saved noise arrays. The editable report files are a **draft**, not a final rendered submission; no final report PDF is included.

## References

1. M. Abdelaty, R. Doriguzzi-Corin, and D. Siracusa, “DAICS: A Deep Learning Solution for Anomaly Detection in Industrial Control Systems,” arXiv:2009.06299v1, 2020. Confirm final publication status and citation details before submission.
2. SWaT dataset/testbed reference: complete bibliographic details must be verified from the cited DAICS paper and the authorized dataset source before publication.
