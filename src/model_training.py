#!/usr/bin/env python3
"""Train three RFMiD disease backbones using a fixed-holdout GridSearchCV.

VS Code terminal (PowerShell):
    python model_training.py --root "C:\\Datasets\\RFMiD" --device auto

GridSearchCV compares 36 configurations without touching official RFMiD test data.
Load the cached preprocessing exports before running this script. On a laptop
without CUDA, full CPU grid search requires explicit --allow-cpu-grid.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import math
import os
from functools import lru_cache
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    Swin_T_Weights, EfficientNet_B0_Weights, DenseNet121_Weights,
    swin_t, efficientnet_b0, densenet121,
)
from torchvision.transforms import InterpolationMode

from preprocessing import DISEASES, CROP_RECIPE, PAD_RECIPE, SIZE, default_root

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
ARCHITECTURES = ('swin_t', 'efficientnet_b0', 'densenet121')
LEARNING_RATES = (1e-5, 3e-5, 1e-4)
PREPROCESSING = ('padded', 'cropped')
AUGMENTATION = ('flip_only', 'strong')


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def atomic_json(data: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')
    temporary.replace(output)


def build_model(architecture: str, pretrained: bool = False) -> nn.Module:
    """Create a 7-label classifier. Checkpoint loading never fetches ImageNet weights."""
    if architecture == 'swin_t':
        model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
        model.head = nn.Linear(model.head.in_features, len(DISEASES))
    elif architecture == 'efficientnet_b0':
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(DISEASES))
    elif architecture == 'densenet121':
        model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier = nn.Linear(model.classifier.in_features, len(DISEASES))
    else:
        raise ValueError(f'Unknown architecture: {architecture}')
    return model


def image_transform(augmentation: str | None = None):
    """Validation/deployment: deterministic; training only: stochastic transforms."""
    ops = []
    if augmentation == 'flip_only':
        ops = [transforms.RandomHorizontalFlip(p=0.5)]
    elif augmentation == 'strong':
        ops = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10, interpolation=InterpolationMode.BICUBIC, fill=0),
            transforms.RandomResizedCrop((SIZE, SIZE), scale=(0.92, 1.0),
                                         ratio=(0.95, 1.05),
                                         interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05),
        ]
    elif augmentation is not None:
        raise ValueError(f'Unknown augmentation: {augmentation}')
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class ImagePathDataset(Dataset):
    def __init__(self, paths, targets=None, transform=None):
        self.paths = list(map(str, paths))
        self.targets = None if targets is None else np.asarray(targets, dtype=np.float32)
        if self.targets is not None and (self.targets.shape != (len(self.paths), len(DISEASES))):
            raise ValueError('Expected a (number_images, 7) label matrix')
        if any(not Path(path).is_file() for path in self.paths):
            raise FileNotFoundError('One or more cached image files are missing; rerun preprocessing.py')
        self.transform = transform or image_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.transform(im.convert('RGB'))
        if self.targets is None:
            return x
        return x, torch.from_numpy(self.targets[i].copy())


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(paths, y, transform, batch_size, workers, seed, shuffle, device):
    return DataLoader(
        ImagePathDataset(paths, y, transform), batch_size=batch_size,
        shuffle=shuffle, num_workers=workers, pin_memory=(device.type == 'cuda'),
        worker_init_fn=seed_worker if workers else None,
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=(workers > 0),
    )


def parse_matrix(x, preprocessing):
    matrix = np.asarray(x, dtype=str)
    if matrix.ndim != 2 or matrix.shape[1] != 2:
        raise ValueError('Expected two path columns in order (padded, cropped)')
    return matrix[:, PREPROCESSING.index(preprocessing)]


def validate_labels(y):
    labels = np.asarray(y)
    if labels.ndim != 2 or labels.shape[1] != len(DISEASES) or not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected binary target array of shape (N, 7)')
    return labels.astype(np.int64)


def score_metrics(y, probabilities):
    y = validate_labels(y)
    p = np.asarray(probabilities, dtype=float)
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid prediction-probability matrix')
    per_class = []
    for j, disease in enumerate(DISEASES):
        positive = int(y[:, j].sum())
        negative = len(y) - positive
        if positive == 0 or negative == 0:
            raise ValueError(f'Cannot evaluate {disease}: needs both positives and negatives')
        per_class.append({
            'disease': disease, 'positives': positive, 'negatives': negative,
            'ap': float(average_precision_score(y[:, j], p[:, j])),
            'auroc': float(roc_auc_score(y[:, j], p[:, j])),
            'f1_at_0_5': float(f1_score(y[:, j], p[:, j] >= 0.5, zero_division=0)),
        })
    rows = pd.DataFrame(per_class)
    return {
        'macro_ap': float(rows.ap.mean()),
        'micro_ap': float(average_precision_score(y.ravel(), p.ravel())),
        'macro_auroc': float(rows.auroc.mean()),
        'macro_f1_at_0_5': float(rows.f1_at_0_5.mean()),
    }, rows


@lru_cache(maxsize=128)
def _check_image_directory(directory: Path, root: Path) -> None:
    """Validate each shared image directory once, not once per image."""
    try:
        directory.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(f'Image directory is outside RFMiD root: {directory}') from exc


def fast_data_path(root: Path, path_value: str) -> Path:
    """Convert a saved relative path to a local path without costly per-file realpath().

    ``root`` is resolved once by main()/run_grid(). Check for ``..`` escapes
    and validate each parent directory once (including Windows junctions).
    """
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f'Invalid cached image path: {path_value!r}')
    path = Path(path_value)
    full = Path(os.path.abspath(str(path if path.is_absolute() else root / path)))
    try:
        full.relative_to(root)
    except ValueError as exc:
        raise ValueError(f'Manifest path outside RFMiD root: {path_value}') from exc
    _check_image_directory(full.parent, root)
    return full


def sha256_training_inputs(x, y, root: Path):
    """Portable provenance: path *relative to RFMiD root*, labels and order.

    This intentionally differs from legacy Colab checkpoints hashed using
    absolute /content/drive paths. Do not blindly reuse legacy checkpoints.
    """
    paths = np.asarray(x, dtype=str)
    paths_rel = [[fast_data_path(root, value).relative_to(root).as_posix()
                  for value in row] for row in paths]
    digest = hashlib.sha256()
    digest.update(json.dumps(paths_rel, separators=(',', ':')).encode('utf-8'))
    digest.update(validate_labels(y).astype(np.uint8).tobytes())
    return digest.hexdigest()


def read_manifests(root: Path):
    project = root / 'revised_supervisor_experiments_v2'
    crop = root / 'processed_datasets' / 'rfmid_min100_cropped224_v2'
    padded = project / 'padded224_grid_control'
    collections = {}
    signatures = {}
    for split in ('train', 'val'):
        info_a = crop / f'{split}_cache_info.json'
        info_b = padded / f'{split}_cache_info.json'
        if not info_a.exists() or not info_b.exists():
            raise FileNotFoundError('Missing preprocessing cache info. Run preprocessing.py first.')
        ca, cb = json.loads(info_a.read_text()), json.loads(info_b.read_text())
        if (ca.get('recipe') != CROP_RECIPE or cb.get('recipe') != PAD_RECIPE or
                ca.get('source_signature') != cb.get('source_signature')):
            raise ValueError('Preprocessing cache recipes/sources are incompatible. Rerun preprocessing.py')
        signatures[split] = ca['source_signature']
        frames = {}
        for key, directory in [('cropped', crop), ('padded', padded)]:
            file = directory / f'{split}_manifest_selected.csv'
            if not file.is_file():
                raise FileNotFoundError(f'{file} missing. Run preprocessing.py first.')
            df = pd.read_csv(file, dtype={'ID': 'string'})
            required = {'ID', 'split', 'source_image_path', 'image_path', *DISEASES}
            if not required.issubset(df.columns):
                raise ValueError(f'{file} missing columns: {required.difference(df.columns)}')
            if df.ID.duplicated().any() or df.ID.isna().any() or not (df.split == split).all():
                raise ValueError(f'{file} contains invalid split or image identifiers')
            if df[DISEASES].isna().any().any() or not df[DISEASES].isin([0, 1]).all().all():
                raise ValueError(f'{file} contains nonbinary labels')
            if df.image_path.isna().any():
                raise FileNotFoundError(f'{file} has missing export image paths; rerun preprocessing.py')
            # Manifest entries are relative to --root, so they survive moving from
            # one Windows drive or laptop folder to another.
            df['image_path'] = df.image_path.map(lambda p: str(fast_data_path(root, p)))
            if not df.image_path.map(lambda p: Path(p).is_file()).all():
                raise FileNotFoundError(f'{file} references missing exported PNGs; rerun preprocessing.py')
            frames[key] = df
        crop_df, pad_df = frames['cropped'], frames['padded']
        if set(crop_df.ID) != set(pad_df.ID):
            raise ValueError(f'Padded and cropped IDs differ: {split}')
        pad_df = pad_df.set_index('ID').loc[crop_df.ID.tolist()].reset_index()
        if not np.array_equal(crop_df[DISEASES].to_numpy(), pad_df[DISEASES].to_numpy()):
            raise ValueError(f'Padded and cropped labels differ: {split}')
        if crop_df.source_image_path.tolist() != pad_df.source_image_path.tolist():
            raise ValueError(f'Padded and cropped original images differ: {split}')
        x = np.column_stack([pad_df.image_path.astype(str), crop_df.image_path.astype(str)])
        y = crop_df[DISEASES].to_numpy(dtype=np.int64)
        collections[split] = (crop_df, x, y)
    train, X_train, y_train = collections['train']
    val, X_val, y_val = collections['val']
    if len(train) != 1920 or len(val) != 640:
        raise ValueError('Expected 1,920 train and 640 validation images')
    # Numeric image IDs may legitimately recur across the official splits;
    # compare canonical source paths and cached export paths instead.
    if set(train.source_image_path) & set(val.source_image_path):
        raise RuntimeError('Train/validation original-image paths overlap')
    for col in (0, 1):
        if set(X_train[:, col]) & set(X_val[:, col]):
            raise RuntimeError('Training and validation images overlap by exported file path')
    # Reload exact ID assignment saved by preprocessing rather than resplitting.
    select_file = project / 'validation_selection_manifest.csv'
    calib_file = project / 'validation_calibration_manifest.csv'
    if not select_file.is_file() or not calib_file.is_file():
        raise FileNotFoundError('Validation selection/calibration manifests missing; run preprocessing.py')
    selection = pd.read_csv(select_file, dtype={'ID': 'string'})
    calibration = pd.read_csv(calib_file, dtype={'ID': 'string'})
    if (set(selection.ID) & set(calibration.ID)) or set(selection.ID) | set(calibration.ID) != set(val.ID):
        raise ValueError('Validation split manifest IDs are missing or overlapping')
    for df in (selection, calibration):
        if not df.ID.isin(val.ID).all() or df.ID.duplicated().any():
            raise ValueError('Invalid validation subset IDs')
        reference = val.set_index('ID').loc[df.ID.tolist()]
        if not np.array_equal(reference[DISEASES].to_numpy(dtype=int), df[DISEASES].to_numpy(dtype=int)):
            raise ValueError('Validation subset labels no longer match processed manifest')
        if any(np.unique(df[DISEASES].to_numpy(dtype=int)[:, j]).size < 2 for j in range(len(DISEASES))):
            raise ValueError('Validation subset contains a disease with only one class')
    ids_to_position = pd.Series(np.arange(len(val)), index=val.ID)
    select_positions = ids_to_position.loc[selection.ID].to_numpy(dtype=int)
    cal_positions = ids_to_position.loc[calibration.ID].to_numpy(dtype=int)
    preprocess_fingerprint = hashlib.sha256(json.dumps({
        'source_signatures': signatures, 'crop_recipe': CROP_RECIPE,
        'pad_recipe': PAD_RECIPE, 'size': SIZE,
    }, sort_keys=True).encode()).hexdigest()
    return {
        'project': project, 'X_train': X_train, 'y_train': y_train,
        'X_selection': X_val[select_positions], 'y_selection': y_val[select_positions],
        'X_calibration': X_val[cal_positions], 'y_calibration': y_val[cal_positions],
        'preprocess_fingerprint': preprocess_fingerprint,
    }


class RFMiDGridEstimator(BaseEstimator):
    """sklearn-compatible estimator. PredefinedSplit ensures fit() sees TRAIN ONLY."""
    def __init__(self, architecture='swin_t', preprocessing='cropped',
                 augmentation='strong', lr=3e-5, epochs=12, weight_decay=0.01,
                 seed=42, batch_size=8, effective_batch_size=16, workers=0,
                 reuse=True, project_dir=None, preprocess_fingerprint=None,
                 device='auto', allow_legacy=False, data_root=None):
        self.architecture = architecture
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.seed = seed
        self.batch_size = batch_size
        self.effective_batch_size = effective_batch_size
        self.workers = workers
        self.reuse = reuse
        self.project_dir = project_dir
        self.preprocess_fingerprint = preprocess_fingerprint
        self.device = device
        self.allow_legacy = allow_legacy
        self.data_root = data_root

    def _device(self):
        if self.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        chosen = torch.device(self.device)
        if chosen.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; check GPU runtime')
        return chosen

    def _directory(self):
        tag = f'{self.architecture}_{self.preprocessing}_{self.augmentation}_lr{self.lr:.0e}'
        return Path(self.project_dir) / 'sklearn_grid_search' / tag

    def _configuration(self, X, y):
        return {
            'architecture': self.architecture, 'preprocessing': self.preprocessing,
            'augmentation': self.augmentation, 'lr': float(self.lr),
            'epochs': int(self.epochs), 'weight_decay': float(self.weight_decay),
            'seed': int(self.seed), 'batch_size': int(self.batch_size),
            'effective_batch_size': int(self.effective_batch_size),
            'disease_order': list(DISEASES), 'num_training_images': len(y),
            'train_fingerprint': sha256_training_inputs(X, y, Path(self.data_root)),
            'preprocess_fingerprint': self.preprocess_fingerprint,
        }

    def fit(self, X, y):
        y = validate_labels(y)
        selected = parse_matrix(X, self.preprocessing)
        if len(y) != len(selected) or len(selected) == 0:
            raise ValueError('Missing or inconsistent fit inputs')
        if self.architecture not in ARCHITECTURES or self.augmentation not in AUGMENTATION:
            raise ValueError('Unsupported model or augmentation')
        if self.epochs < 1 or self.effective_batch_size < self.batch_size or self.effective_batch_size % self.batch_size:
            raise ValueError('epochs >= 1 and effective batch size must divide cleanly by batch_size')
        self.classes_ = np.arange(len(DISEASES))
        self.directory_ = str(self._directory())
        folder = Path(self.directory_)
        folder.mkdir(parents=True, exist_ok=True)
        self.checkpoint_ = str(folder / 'best.pt')
        self.config_ = self._configuration(X, y)
        self.model_ = None
        self.device_ = self._device()
        checkpoint = Path(self.checkpoint_)
        if checkpoint.is_file():
            if not self.reuse:
                raise FileExistsError(f'Checkpoint already exists; use another output location: {checkpoint}')
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            original = state.get('grid_config', {})
            if original != self.config_:
                old_without_preprocess = {k: v for k, v in self.config_.items() if k != 'preprocess_fingerprint'}
                if not (self.allow_legacy and original == old_without_preprocess):
                    raise ValueError(f'Checkpoint config mismatch: {checkpoint}. '
                                     'Preserve existing checkpoints and use a new --experiment-name or '
                                     'restore original settings. Legacy checkpoints require --allow-legacy.')
                print('WARNING: legacy checkpoint has no preprocessing fingerprint; verify its provenance manually.')
            if state.get('target_order') != DISEASES:
                raise ValueError('Cached checkpoint disease order mismatch')
            print('[CHECKPOINT REUSED]', checkpoint)
            return self
        if any(not Path(p).is_file() for p in selected):
            raise FileNotFoundError('A training image is missing; rerun preprocessing.py')
        set_seed(self.seed)
        self.model_ = build_model(self.architecture, pretrained=True).to(self.device_)
        dl = make_loader(selected, y, image_transform(self.augmentation),
                         self.batch_size, self.workers, self.seed, True, self.device_)
        pos = y.sum(axis=0)
        neg = len(y) - pos
        if np.any(pos <= 0) or np.any(neg <= 0):
            raise ValueError('Train partition requires positive and negative examples for each disease')
        weights = torch.tensor(neg / pos, dtype=torch.float32, device=self.device_)
        objective = nn.BCEWithLogitsLoss(pos_weight=weights)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scaler = torch.amp.GradScaler('cuda', enabled=(self.device_.type == 'cuda'))
        accumulation = self.effective_batch_size // self.batch_size
        history = []
        try:
            for epoch in range(1, self.epochs + 1):
                self.model_.train()
                total, count = 0.0, 0
                optimizer.zero_grad(set_to_none=True)
                # Correct scaling for the possibly shorter final accumulation group.
                for batch_index, (x, target) in enumerate(dl):
                    x = x.to(self.device_, non_blocking=True)
                    target = target.to(self.device_, non_blocking=True)
                    remaining = len(dl) - (batch_index // accumulation) * accumulation
                    group_size = min(accumulation, remaining)
                    context = torch.autocast('cuda', dtype=torch.float16) if self.device_.type == 'cuda' else nullcontext()
                    with context:
                        unscaled_loss = objective(self.model_(x), target)
                        loss = unscaled_loss / group_size
                    if not bool(torch.isfinite(loss).item()):
                        raise FloatingPointError(f'Non-finite loss at epoch {epoch}')
                    scaler.scale(loss).backward()
                    if (batch_index + 1) % accumulation == 0 or (batch_index + 1) == len(dl):
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                    total += float(unscaled_loss.item()) * len(x)
                    count += len(x)
                row = {'epoch': epoch, 'train_loss': total / count}
                history.append(row)
                print(f'{folder.name}: epoch {epoch}/{self.epochs} train loss {row["train_loss"]:.5f}', flush=True)
            pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
            # Keep the original notebook's checkpoint format, so prior deployment works.
            temporary = checkpoint.with_suffix('.tmp')
            torch.save({'model_state': self.model_.state_dict(), 'epoch': int(self.epochs),
                        'target_order': list(DISEASES), 'grid_config': self.config_}, temporary)
            temporary.replace(checkpoint)
            atomic_json(self.config_, folder / 'grid_config.json')
        finally:
            del dl, optimizer, objective
        return self

    def predict_proba(self, X):
        if not hasattr(self, 'checkpoint_'):
            raise RuntimeError('fit() must be called before predict_proba()')
        paths = parse_matrix(X, self.preprocessing)
        if self.model_ is None:
            checkpoint = torch.load(self.checkpoint_, map_location='cpu', weights_only=True)
            if checkpoint.get('target_order') != DISEASES:
                raise ValueError('Checkpoint label order does not match')
            self.model_ = build_model(self.architecture, pretrained=False).to(self.device_)
            self.model_.load_state_dict(checkpoint['model_state'], strict=True)
        self.model_.eval()
        dl = make_loader(paths, None, image_transform(), self.batch_size, self.workers,
                         self.seed, False, self.device_)
        predictions = []
        with torch.inference_mode():
            for x in dl:
                logits = self.model_(x.to(self.device_, non_blocking=True))
                predictions.append(torch.sigmoid(logits).float().cpu().numpy())
        if not predictions:
            raise ValueError('Cannot predict an empty validation dataset')
        return np.concatenate(predictions, axis=0)


def selection_macro_ap(estimator, X, y_true):
    true = validate_labels(y_true)
    predicted = estimator.predict_proba(X)
    summary, per_class = score_metrics(true, predicted)
    folder = Path(estimator.directory_)
    np.savez_compressed(folder / 'selection_predictions.npz', y=true, probabilities=predicted)
    per_class.to_csv(folder / 'selection_per_class_metrics.csv', index=False)
    result = {
        'architecture': estimator.architecture,
        'preprocessing': estimator.preprocessing,
        'augmentation': estimator.augmentation,
        'lr': float(estimator.lr),
        'best_epoch': int(estimator.epochs),
        'checkpoint': estimator.checkpoint_,
        'selection_predictions': str(folder / 'selection_predictions.npz'),
        **summary,
    }
    atomic_json(result, folder / 'results.json')
    estimator.model_ = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary['macro_ap']


def verify_preprocessing_signature(root: Path, preprocess_fingerprint: str) -> None:
    """Ensure the two saved cache metadata files are still present and compatible."""
    data = read_manifests(root)
    if data['preprocess_fingerprint'] != preprocess_fingerprint:
        raise RuntimeError('Preprocessing cache metadata changed during training')


def run_grid(root: Path, epochs=12, batch_size=8, effective_batch_size=16,
             workers=0, device='auto', reuse=True, allow_legacy=False, allow_cpu_grid=False,
             experiment_name=None, architectures=None, learning_rates=None,
             preprocessings=None, augmentations=None):
    data = read_manifests(root)
    project = data['project'] if experiment_name is None else data['project'] / experiment_name
    project.mkdir(parents=True, exist_ok=True)
    X_tr, y_tr = data['X_train'], data['y_train']
    X_sel, y_sel = data['X_selection'], data['y_selection']
    X = np.concatenate((X_tr, X_sel), axis=0)
    y = np.concatenate((y_tr, y_sel), axis=0)
    fold = PredefinedSplit(np.r_[np.full(len(X_tr), -1), np.zeros(len(X_sel), dtype=int)])
    fit_idx, select_idx = next(fold.split(X, y))
    assert len(fit_idx) == len(X_tr) and len(select_idx) == len(X_sel)
    assert np.array_equal(fit_idx, np.arange(len(X_tr)))
    assert np.array_equal(select_idx, np.arange(len(X_tr), len(X)))
    assert len(data['X_calibration']) + len(X_sel) == 640
    space = {
        'architecture': list(architectures or ARCHITECTURES),
        'preprocessing': list(preprocessings or PREPROCESSING),
        'augmentation': list(augmentations or AUGMENTATION),
        'lr': list(learning_rates or LEARNING_RATES),
    }
    count = math.prod(len(v) for v in space.values())
    actual_device = ('cuda' if device == 'auto' and torch.cuda.is_available() else
                     'cpu' if device == 'auto' else device)
    print(f'Training device: {actual_device}; DataLoader workers: {workers}')
    if actual_device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable. Check NVIDIA GPU and PyTorch CUDA install.')
        print(f'NVIDIA GPU: {torch.cuda.get_device_name(0)}')
    elif not allow_cpu_grid:
        raise RuntimeError('No CUDA GPU detected. A 36-configuration CPU grid can take days. '
                           'Install CUDA-enabled PyTorch if an NVIDIA GPU is present, '
                           'or explicitly add --allow-cpu-grid if you accept CPU training.')
    else:
        print('WARNING: CPU training may be extremely slow.')
    print(f'Config count: {count}; training={len(X_tr)}, selection={len(X_sel)}, '
          f'calibration={len(data["X_calibration"])}; official test is NOT loaded.')
    estimator = RFMiDGridEstimator(
        project_dir=str(project), preprocess_fingerprint=data['preprocess_fingerprint'],
        data_root=str(root),
        epochs=epochs, batch_size=batch_size, effective_batch_size=effective_batch_size,
        workers=workers, device=device, reuse=reuse, allow_legacy=allow_legacy,
    )
    grid = GridSearchCV(estimator, space, cv=fold, scoring=selection_macro_ap,
                        refit=False, n_jobs=1, verbose=2, error_score='raise')
    grid.fit(X, y)
    table = pd.DataFrame(grid.cv_results_)
    table.to_csv(project / 'gridsearchcv_full_results.csv', index=False)
    records = []
    for file in (project / 'sklearn_grid_search').glob('*/results.json'):
        record = json.loads(file.read_text(encoding='utf-8'))
        # Only include runs in the requested search, not unrelated stale trials.
        if (record['architecture'] in space['architecture'] and
            record['preprocessing'] in space['preprocessing'] and
            record['augmentation'] in space['augmentation'] and
            record['lr'] in space['lr']):
            records.append(record)
    results = pd.DataFrame(records)
    results.to_csv(project / 'all_preprocessing_augmentation_lr_results.csv', index=False)
    fair = results.loc[(results.preprocessing == 'cropped') &
                       (results.augmentation == 'strong')].copy()
    fair.to_csv(project / 'all_architecture_lr_results.csv', index=False)
    winners = (fair.sort_values(['architecture', 'macro_ap', 'lr'],
                               ascending=[True, False, True])
               .drop_duplicates('architecture'))
    # Write backbone winners only when each architecture's entire 3-LR comparison ran.
    expected = set(space['architecture'])
    complete = all(len(fair.loc[fair.architecture == a]) == len(space['lr']) for a in expected)
    if len(expected) == 3 and complete and len(winners) == 3:
        winners.to_csv(project / 'backbone_winners.csv', index=False)
    else:
        print('Partial grid: backbone_winners.csv not updated; do not deploy as full comparison.')
    # Overall winner across all 36, including padded and flip-only configurations.
    best_config = dict(grid.best_params_)
    best_config['validation_macro_ap'] = float(grid.best_score_)
    tag = f'{best_config["architecture"]}_{best_config["preprocessing"]}_{best_config["augmentation"]}_lr{best_config["lr"]:.0e}'
    best_config['checkpoint'] = str(project / 'sklearn_grid_search' / tag / 'best.pt')
    best_config['preprocess_fingerprint'] = data['preprocess_fingerprint']
    atomic_json(best_config, project / 'best_grid_configuration.json')
    print('Best selection macro AP:', grid.best_score_)
    print('Best parameters:', grid.best_params_)
    print('Selected checkpoint:', best_config['checkpoint'])
    return grid, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=default_root(),
                        help='Local folder containing Training_Set and Evaluation_Set')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--effective-batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0,
                        help='0 avoids notebook multiprocessing-worker shutdown errors')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--allow-cpu-grid', action='store_true',
                        help='Explicitly acknowledge that full training on CPU can be extremely slow')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Use a NEW subfolder when deliberately changing training settings')
    parser.add_argument('--no-reuse', action='store_true')
    parser.add_argument('--allow-legacy', action='store_true',
                        help='Only if you audited older checkpoints lacking source fingerprints')
    parser.add_argument('--smoke', action='store_true',
                        help='One architecture/configuration at 1 epoch, NOT for research claims')
    args = parser.parse_args()
    if args.smoke:
        run_grid(args.root.expanduser().resolve(), epochs=1, batch_size=args.batch_size,
                 effective_batch_size=args.effective_batch_size, workers=args.num_workers,
                 device=args.device, reuse=True, allow_cpu_grid=True, experiment_name='smoke_test',
                 architectures=['efficientnet_b0'], learning_rates=[3e-5],
                 preprocessings=['cropped'], augmentations=['flip_only'])
    else:
        run_grid(args.root.expanduser().resolve(), epochs=args.epochs,
                 batch_size=args.batch_size, effective_batch_size=args.effective_batch_size,
                 workers=args.num_workers, device=args.device, reuse=not args.no_reuse,
                 allow_cpu_grid=args.allow_cpu_grid,
                 allow_legacy=args.allow_legacy, experiment_name=args.experiment_name)


if __name__ == '__main__':
    main()
