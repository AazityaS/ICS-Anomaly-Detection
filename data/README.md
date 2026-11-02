# SWaT data setup

The SWaT dataset is not included in this repository. Obtain it only from an authorized dataset provider or through institutional access, and comply with its terms of use. The repository does not redistribute raw or processed SWaT records.

## Expected local files

Place the supplied CSV files under `data/SWaT/` using these names:

- `normal.csv`
- `attack.csv`
- `merged.csv`

The preprocessing script currently expects all three files. `merged.csv` is used for preprocessing/reporting but is not treated as an independent test dataset. Normal and attack sources are kept separate for temporal splitting and source-local test windows.

## Run preprocessing

From the project root with Python 3.12 and project dependencies installed:

```powershell
python preprocess_swat.py
```

The script writes `data/processed/normal_clean.csv`, `attack_clean.csv`, `merged_clean.csv`, and `preprocessing_report.json`. It removes the seven unavailable variables (`MV101`, `AIT201`, `MV201`, `P201`, `P202`, `P204`, `MV303`), removes exact duplicate rows, parses timestamps, sorts by time, and checks missing values and duplicate timestamps. It overwrites those output paths, so preserve any local artifacts you need before rerunning it.

The cleaned operational dataset has 44 variables (24 sensors and 20 actuators) plus timestamp and label. Sensors are normalized using training-only fit parameters; actuator states remain discrete and unscaled. Training and validation use normal data only. Test windows are generated independently for the remaining normal source and the attack source. The calendar intervals overlap across source files; attack-source rows are not used for training or validation.

Do not stage or commit any CSV from `data/SWaT/` or the processed CSV files. Only the small preprocessing report is whitelisted for repository publication.
