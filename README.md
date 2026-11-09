# MAPPER-Scripts

Analysis scripts for **MAPPER** (Multidimensional Analysis of Personality Prediction from Electroencephalographic Representations), accompanying the MAPPER EEG–MCMI-IV manuscript.

## Contents

The pipeline is organized into eight processing/analysis stages and a GUI that can be used to navigate the pipeline:

| File | Purpose |
| --- | --- |
| `0_project_config_V3.py` | Configure and run the pipeline (Main GUI) |
| `1_preprocess_basic_V3.py` | Basic EEG preprocessing |
| `2_preprocess_epochs_V3.py` | Epoch preprocessing and artifact rejection |
| `3_preprocessing_qc_report_V3.py` | Preprocessing quality control |
| `4_build_feature_cache_V3.py` | EEG feature extraction and caching |
| `5_create_fixed_outer_folds_V3.py` | Fixed outer cross-validation folds |
| `6_nested_mcmi_V3.py` | Nested MCMI analyses |
| `7_nested_results_report_V3.py` | Results summary and reporting |
| `8_nested_inference_V3.py` | Statistical inference |

The five supporting modules are `neuropsyche_v3_config.py`, `V3_autoreject_worker.py`, `V3_config_loader.py`, `V3_model_core.py`, and `V3_runtime.py`.

## Environment and use

The recorded analysis environment was **Python 3.11.14**. `requirements.txt` lists the directly imported third-party packages with their recorded versions. The GUI requires Tkinter.

In the GUI, configure the local EEG/assessment input paths and output folders, then select and run the needed stages. The input EEG format configured by default is EDF. The code expects participant identifiers to connect EEG recordings with the assessment table.

`V3_user_settings.json` is the included GUI settings file; it has blank data paths. Its values are not a declaration that every reported analysis used those settings. Run-specific manifests and settings snapshots record the effective settings of individual runs. Refer to the accompanying manuscript for the reported analyses and methodological details.

**Data availability:** This repository contains scripts and configuration only; it does not contain participant-level EEG or assessment data.
