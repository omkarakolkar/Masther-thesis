# Master Thesis Project

This repository contains the code and experimental setup for a master thesis project focused on sound event detection (SED) using synthetic mixtures and a CRNN-based baseline model.

The project is organized into two main parts:

- Data generation workflow for creating synthetic audio mixtures from isolated events and background noise.
- A baseline SED pipeline (`FDY_SED_baseline`) configured for weakly/strongly supervised training with a CRNN architecture.

## Project overview

The work is centered on generating realistic acoustic mixtures and training a sound event detection system to localize and classify events in 10-second clips. The pipeline includes:

- isolated event loading and filtering
- background noise sampling
- synthetic mixture generation with event placement and labeling
- dataset splitting and label extraction
- baseline training configuration for a CRNN-based event detector

## Repository structure

```text
.
├── Data_generation/
│   ├── 1_CSV_filtering&copying_files.py
│   ├── 2.1_mixture_generation.py
│   ├── 2.2_mixture_gen_nonOverlapp_multi_events.py
│   ├── 3_split_generation.py
│   └── extract_single_labels.py
│
├── FDY_SED_baseline/
│   └── configs/
│       ├── config.yaml
│       ├── config_server.yaml
│       └── utils/
│
├── Main_Master_Thesis_Report.pdf
├── Master_Thesis_Presentation .pdf
├── README.md
└── .gitignore
```

## Data generation

The `Data_generation/` folder contains the scripts used to generate synthetic mixtures and dataset splits.

### Main scripts

- `1_CSV_filtering&copying_files.py`
  - Filters and organizes CSV metadata and associated audio files.

- `2.1_mixture_generation.py`
  - Generates audio mixtures by combining isolated event clips with background noise while creating metadata labels.

- `2.2_mixture_gen_nonOverlapp_multi_events.py`
  - Generates additional mixtures with multi-event, non-overlapping patterns.

- `3_split_generation.py`
  - Creates train/validation/test dataset splits.

- `extract_single_labels.py`
  - Extracts single-label information from generated data.

These scripts assume local dataset directories and audio files on disk, and some paths are hard-coded to machine-specific locations.

## Baseline model

The `FDY_SED_baseline/` directory contains the training configuration for a CRNN-based sound event detector. The configuration file (`FDY_SED_baseline/configs/config.yaml`) defines:

- audio feature extraction parameters
- training hyperparameters
- CRNN architecture settings
- dataset paths for weak, unlabeled, test, and synthetic data
- normalization and augmentation options

The configuration is tuned for a multi-class SED setup and includes augmentation strategies such as mixup, filtering, frequency masking, and noise injection.

## Prerequisites

This project is built around Python and audio processing libraries. Typical dependencies include:

- Python 3.9+
- PyTorch
- Librosa
- SoundFile
- NumPy
- tqdm
- PyYAML

## Setup

1. Clone the repository.
2. Install the required dependencies.
3. Update dataset paths in `FDY_SED_baseline/configs/config.yaml` to match your local environment.
4. Prepare your isolated events, background audio, and label files.
5. Run the generation scripts in `Data_generation/` as needed.
6. Train and evaluate the baseline model using the configured pipeline in `FDY_SED_baseline`.

## Notes

- The dataset paths in the project configuration currently point to local `/home/...` directories.
- The repository includes both the thesis report and presentation PDF files for reference.
- This is a research-oriented codebase, so path and environment configuration may need adjustment depending on the machine or dataset layout used.

## Related files

- `Main_Master_Thesis_Report.pdf` — thesis report
- `Master_Thesis_Presentation .pdf` — presentation slides

## License

This repository does not currently declare a license file. Please check with the repository owner before reusing the code or materials for publication or distribution.

## Contact

For questions about the project or repository structure, please contact the repository owner or refer to the thesis materials included in this project.
