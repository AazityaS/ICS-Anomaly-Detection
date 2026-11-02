# Curated publication results

This directory contains compact copies of saved project outputs and only the verified model/calibration artifacts used by the reported clean evaluations. Source experiment files were not modified. No training, calibration, or evaluation was run to prepare this directory.

- `wdnn_clean_evaluation.json`, `gru_clean_evaluation.json`, and the clean comparison CSV contain source and pooled evaluation details.
- Robustness CSVs preserve each model's saved noise-condition metrics and comparison. The original noise arrays and point-level prediction dumps are intentionally excluded.
- `preprocessing_report.json` is copied from the preprocessing output. No dataset records are included.
- `checkpoints/` contains WDNN epoch 100, the corresponding WDNN TTNN/threshold/scaler/actuator database, and GRU best epoch 12 with its own TTNN and thresholds. `checkpoints/provenance.json` records source paths, evaluation use, epochs, and exclusions. The WDNN epoch-90 best-validation weights were not saved.
- `figures/` contains selected architecture, training, clean metric, robustness, and attribution figures copied from `report/figures/`.

The GRU TTNN next-MSE target remains an approximation. Thresholds are model-specific. The models are separate; GRU is not integrated into DAICS. These outputs do not constitute a complete reproduction of the original DAICS paper.
