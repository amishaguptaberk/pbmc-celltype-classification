# BioE 145 Final Project

## Structure
- `data/` — processed_counts.csv and labels.csv
- `part1.ipynb` — Autoencoder + PCA/t-SNE visualizations
- `part2.ipynb` — Cell type classifier
- `requirements.txt` — Python dependencies

## Setup
```bash
pip install -r requirements.txt
jupyter notebook
```

## Data Notes
- 700 cells × 765 genes (log-normalized scRNA-seq)
- Label column: `bulk_labels` (not "cell type")
- 10 PBMC cell types
