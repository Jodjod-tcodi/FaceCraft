# FaceCraft

FaceCraft is a deep learning project for text-to-face synthesis and facial image editing using CelebA, CelebA-Dialog, and Face2Text datasets.

## Project Structure

```text
FaceCraft/
│
├── data/
│   ├── raw/
│   │   ├── celeba/
│   │   ├── celeba_dialog/
│   │   └── face2text/
│   │
│   ├── processed/
│   │   ├── images/
│   │   ├── captions.csv
│   │   └── metadata.csv
│   │
│   └── samples/
│
├── notebooks/
│   └── Deliverable_2_Preprocessing.ipynb
│
├── models/
│
└── README.md
```

## Directory Descriptions

- **`data/raw/`**: Raw datasets (`celeba`, `celeba_dialog`, `face2text`).
- **`data/processed/`**: Cleaned and formatted images, aligned captions (`captions.csv`), and metadata (`metadata.csv`).
- **`data/samples/`**: Sample images and intermediate outputs for quick verification.
- **`notebooks/`**: Jupyter notebooks for exploratory data analysis, preprocessing pipelines, and experiments.
- **`models/`**: Saved model checkpoints and weights.

