# EyeQ quality-model integration for RFMiD — VS Code / Windows

**Research goal:** Train a real three-class quality classifier (`Good`, `Usable`, `Reject`) on EyeQ, then pass its quality predictions to the RFMiD M2 quality-aware fusion model and M3 calibrated referral system. This is a research prototype, not a clinical screening tool.

The bundled `model_deployment.py`, `model_training.py` and `preprocessing.py` are the compatible **M1–M3 research scripts**. This package adds **`eyeq_quality.py`**. It does not retrain the user's already-selected M1 Swin-T or repeat GridSearchCV.

## Data acquisition and scientific boundaries

Official EyeQ: <https://github.com/HzFu/EyeQ>.

- The official repository provides **quality annotations** in `data/Label_EyeQ_train.csv` (among others). For the source annotations, `quality=0` corresponds to **good**, `1` to **usable**, and `2` to **reject**.
- The corresponding **fundus photographs are not in the EyeQ GitHub repository**. Obtain permitted original EyePACS photographs separately and match filenames to the quality annotations. Ensure you follow applicable dataset terms.
- `eyeq_quality.py prepare` intentionally accepts the official **EyeQ training CSV only**, and splits it into *patient-disjoint* train and validation subsets. It never reads official EyeQ `Label_EyeQ_test.csv`.
- Do not interpret the resulting EyeQ **validation** metrics or RFMiD **selection** metrics as independent final-test metrics. Do not access official RFMiD test images during M1/M2/M3 model or policy development.

## Folder layout

```text
RFMiD_VSCode_Local/
├── eyeq_quality.py              <-- NEW
├── model_deployment.py          <-- M1/M2/M3 research completion version
├── model_training.py
├── preprocessing.py
├── requirements.txt
├── EyeQ/
│   ├── Label_EyeQ_train.csv     <-- REAL official annotation file
│   ├── images/                   <-- REAL EyePACS fundus image files
│   └── eyeq_manifest.csv         <-- generated
└── RFMiD/
    └── revised_supervisor_experiments_v2/
        ├── sklearn_grid_search/
        │   └── swin_t_cropped_flip_only_lr1e-04/
        │       └── best.pt
        └── m1_m2_m3_research/    <-- EyeQ + M2/M3 outputs
```

## 0. Install requirements (using your .venv environment)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In the examples below, commands assume that PowerShell is in the directory containing the four scripts. For PyTorch, use the matching official CPU/GPU distribution for your environment. A laptop with Intel HD Graphics 630 / AMD Radeon R5 430 and CPU-only PyTorch is not CUDA-capable; use a remote NVIDIA GPU if training takes too long.

## 1. Build a real EyeQ training/validation manifest from original training annotations

```powershell
python eyeq_quality.py prepare `
    --labels ".\EyeQ\Label_EyeQ_train.csv" `
    --images ".\EyeQ\images" `
    --output ".\EyeQ\eyeq_manifest.csv"
```

This indexes actual image paths once and saves `image_path`, `quality_label`, `patient_id` and `split` columns. It matches official filename or (if unique) image filename stem, checks missing/ambiguous images and uses patient-ID-aware stratified grouping. The official test annotations are **never** used.

If only a subset of EyeQ images has been downloaded, `--allow-missing` explicitly permits a **partial pilot**, but it does **not** make that pilot a full-dataset evaluation. The script still requires enough examples in every quality category.

## 2. Train EyeQ EfficientNet-B0 and export checkpoint for M2/M3

```powershell
python eyeq_quality.py train `
    --manifest ".\EyeQ\eyeq_manifest.csv" `
    --output ".\RFMiD\revised_supervisor_experiments_v2\m1_m2_m3_research" `
    --epochs 12 --batch-size 16 --lr 0.0001 `
    --device auto --workers 0
```

What happens:

- Loads ImageNet-pretrained EfficientNet-B0, replaces final classification layer with **3 logits**.
- Freezes the visual backbone by default and fits the quality classification head with class-weighted cross-entropy; optional `--fine-tune` unfreezes the final two feature blocks.
- Uses **horizontal flips for training only**, and consistent resizing/padding to 224×224 with ImageNet normalization; avoids brightness/blur augmentation that could alter quality labels.
- Selects the best epoch by *EyeQ validation macro F1*, uses early stopping and saves a compatible **`eyeq_quality.pt`** file.
- On rerun with the same manifest and arguments, reuses the checkpoint instead of retraining. Different settings/data require a new output folder or explicit `--force`.
- Writes `eyeq_history.csv`, `eyeq_quality_validation_metrics.json`, `eyeq_validation_predictions.csv`, and `eyeq_confusion_matrix.png`.

Use `--fine-tune` if head-only training plateaus, ideally on a remote GPU. The included `--no-pretrained` flag is for code smoke-testing **only**, not for scientific results.

**Compatibility:** The code deliberately uses the exact `QualityNet` definition and preprocessing from the bundled `model_deployment.py`/`model_training.py`. The saved checkpoint has keys `model_state`, `class_order`, `eyeq_signature` etc. and can be read by `model_deployment.load_quality()`.

## 3. Review the EyeQ validation metrics

```powershell
python eyeq_quality.py evaluate `
    --manifest ".\EyeQ\eyeq_manifest.csv" `
    --checkpoint ".\RFMiD\revised_supervisor_experiments_v2\m1_m2_m3_research\eyeq_quality.pt" `
    --output ".\RFMiD\revised_supervisor_experiments_v2\m1_m2_m3_research\eyeq_eval" `
    --device auto
```

Metrics: validation macro-F1, balanced accuracy, accuracy, per-class precision/recall/F1 and 3×3 confusion matrix. They measure EyeQ quality classification, **not** seven-disease RFMiD screening effectiveness.

## 4. Use EyeQ in M2 and M3 (existing Swin-T GridSearchCV winner)

```powershell
python model_deployment.py fit `
    --root ".\RFMiD" `
    --quality-checkpoint ".\RFMiD\revised_supervisor_experiments_v2\m1_m2_m3_research\eyeq_quality.pt" `
    --checkpoint ".\RFMiD\revised_supervisor_experiments_v2\sklearn_grid_search\swin_t_cropped_flip_only_lr1e-04\best.pt" `
    --device auto --num-workers 0
```

Then validate the resulting M1/M2/M3 systems on **RFMiD development/selection data**:

```powershell
python model_deployment.py evaluate --root ".\RFMiD" --device auto
```

**Do not run `model_deployment.py final-test` until all experiments, thresholds and referral policy have been finalized and frozen.**

## 5. Test an external retinal photograph with EyeQ alone

```powershell
python eyeq_quality.py predict `
    --checkpoint ".\RFMiD\revised_supervisor_experiments_v2\m1_m2_m3_research\eyeq_quality.pt" `
    --image "C:\Images\external_fundus.jpg" `
    --device auto
```

It returns predicted quality class and three softmax probabilities.

## Reproducibility and limitations

- EyeQ and RFMiD are separate datasets; never treat EyeQ's image-quality labels as disease ground truth.
- The EyeQ/quality validation split is **patient disjoint** by the standard `<patient>_left/right` filename grouping. If your actual dataset has different naming or patient identifiers, supply a correct `patient_id` column or adapt the mapping accordingly.
- Real EyeQ training on actual images **has not been executed** by this code package. Synthetic-data smoke tests demonstrate that code paths and checkpoint compatibility work; they are not research metrics.
- Head-only fine-tuning is a transfer-learning baseline; no guaranteed clinical reliability or improvement over M1.
