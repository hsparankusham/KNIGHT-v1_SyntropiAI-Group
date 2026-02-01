# KNIGHT v1

**Brain Neuroimmune Cell State Foundation Model**

*Syntropi AI Group*

---

KNIGHT (**K**nowledge-guided **N**euroimmune **I**ntelli**g**ent **H**igh-**T**hroughput) is a foundation model for brain neuroimmune cell state classification, cross-dataset harmonization, and perturbation response prediction.

## What KNIGHT Does

1. **Cell State Classification** — Classify brain immune cells into 40+ fine-grained states (DAM stages, inflammatory microglia, reactive astrocyte subtypes) from scRNA-seq profiles with ≥90% balanced accuracy
2. **Cross-Dataset Harmonization** — Integrate single-cell data across labs, platforms, and cohorts with ≤10% batch effect residual
3. **Perturbation Prediction** — Predict gene expression changes following CRISPRi knockdowns or drug treatments (≥0.6 Pearson correlation)

## Architecture

KNIGHT uses a transformer encoder pretrained with masked gene expression modeling (analogous to BERT's masked language modeling, but for gene expression values). The architecture:

- **Encoder**: Gene token embeddings + continuous expression value encoding → transformer layers → cell embedding
- **KNIGHT-min** (prototype): 100M parameters, 6 layers, 256 hidden dim, 4 attention heads
- **KNIGHT-full** (target): 500M+ parameters, 12 layers, 512 hidden dim, 8 attention heads
- **Task heads**: Cell state classifier (hierarchical), perturbation predictor (cross-attention)

## Data Sources

| Source | Type | Cells | Use |
|--------|------|-------|-----|
| SEA-AD Atlas | scRNA-seq + scATAC-seq | ~5M | Primary pretraining |
| Allen Brain Atlas | scRNA-seq | ~2M | Pretraining diversity |
| GEO (8 datasets) | scRNA-seq | ~3M | Cross-dataset validation |
| CRISPRi screens | Perturb-seq | ~500K | Perturbation fine-tuning |

## Project Structure

```
KNIGHT-v1_SyntropiAI-Group/
├── knight/                    # Core Python package
│   ├── data/                  # Data loading & preprocessing
│   │   ├── downloaders/       # Dataset download scripts
│   │   ├── preprocessors/     # QC, normalization, HVG selection
│   │   ├── harmonizer/        # Batch correction (scVI, Harmony)
│   │   └── datasets/          # PyTorch Dataset classes
│   ├── model/                 # Model architecture
│   │   ├── architectures/     # KnightEncoder, KnightModel
│   │   ├── layers/            # Custom layers, tokenizer
│   │   └── heads/             # Task-specific heads
│   ├── training/              # Training loops
│   ├── evaluation/            # Benchmarks & metrics
│   ├── inference/             # Inference engine
│   └── utils/                 # Logging, I/O, reproducibility
├── config/                    # YAML configurations
├── scripts/                   # Pipeline entry points
│   ├── data_curation/         # Steps 01-07: download → splits
│   ├── training/              # Steps T1-T3: pretrain → finetune
│   └── evaluation/            # Steps E1-E3: benchmark suite
├── data/                      # Data directory (gitignored)
├── models/                    # Checkpoints (gitignored)
├── notebooks/                 # Exploratory notebooks
├── tests/                     # Unit & integration tests
└── docs/                      # Documentation
```

## Quick Start

```bash
# Install
pip install -e ".[dev,notebooks,data]"

# Download data
make data-download

# Preprocess
make preprocess

# Train KNIGHT-min prototype
make pretrain

# Evaluate
make evaluate
```

## Pipeline

### Data Curation (Steps 01–07)
1. Download SEA-AD atlas
2. Download GEO datasets
3. Download Allen Brain Atlas
4. Preprocess expression (QC → normalize → HVG → PCA)
5. Preprocess chromatin (peaks → TF-IDF → LSI)
6. Harmonize batches (scVI integration)
7. Create donor-stratified train/val/test splits

### Training (Steps T1–T3)
1. **Pretrain** KNIGHT-min with masked expression modeling
2. **Fine-tune** cell state classification head
3. **Fine-tune** perturbation prediction head

### Evaluation (Steps E1–E3)
1. Cell state classification benchmark (target: ≥90% balanced accuracy)
2. Perturbation prediction benchmark (target: ≥0.6 Pearson)
3. Harmonization benchmark (target: ≤10% batch residual)

## Performance Targets

| Task | Metric | Target | Comparison |
|------|--------|--------|------------|
| Cell state classification | Balanced accuracy | ≥0.90 | scGPT: ~0.85 |
| Batch harmonization | Batch effect residual | ≤0.10 | scVI: ~0.15 |
| Perturbation prediction | Pearson correlation | ≥0.60 | GEARS: ~0.50 |

## License

MIT
