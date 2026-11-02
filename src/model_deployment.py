#!/usr/bin/env python3
"""RFMiD research completion: M1 baseline, M2 quality fusion, M3 uncertainty/referral.

Run from the folder holding preprocessing.py and model_training.py.

Commands:
  python model_deployment.py fit --root RFMiD --eyeq-manifest eyeq_manifest.csv
  python model_deployment.py evaluate --root RFMiD
  python model_deployment.py predict --root RFMiD --image image.jpg
  python model_deployment.py final-test --root RFMiD --test-dir PATH --test-csv PATH \
      --unlock-final-test I_CONFIRM_FROZEN_PROTOCOL

The official RFMiD test partition is never loaded by fit/evaluate/predict.
This is a research prototype, not a medical device.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from preprocessing import DISEASES, EXTENSIONS, default_root, retina_bbox_crop, resize_pad
from model_training import build_model, image_transform, read_manifests, set_seed

QUALITY_NAMES = ['good', 'usable', 'reject']
MODEL_DIR_NAME = 'm1_m2_m3_research'
SEED = 42


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
    temp.replace(path)


def torch_save_atomic(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pt.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def digest_file(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def short_path_hash(paths):
    return hashlib.sha256(json.dumps([str(x) for x in paths], separators=(',', ':')).encode()).hexdigest()


def choose_device(value):
    if value == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but CUDA is unavailable on this machine.')
    return torch.device('cuda' if value == 'auto' and torch.cuda.is_available()
                        else 'cpu' if value == 'auto' else value)


def safe_torch_load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def find_m1_checkpoint(root: Path, explicit: Path | None):
    if explicit is not None:
        checkpoint = explicit.expanduser().resolve()
    else:
        config_file = root / 'revised_supervisor_experiments_v2' / 'best_grid_configuration.json'
        if not config_file.is_file():
            raise FileNotFoundError(f'GridSearchCV winner not found: {config_file}; use --checkpoint')
        selected = json.loads(config_file.read_text(encoding='utf-8'))
        tag = (f"{selected['architecture']}_{selected['preprocessing']}_"
               f"{selected['augmentation']}_lr{float(selected['lr']):.0e}")
        checkpoint = config_file.parent / 'sklearn_grid_search' / tag / 'best.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'M1 checkpoint does not exist: {checkpoint}')
    return checkpoint


def load_m1(checkpoint_path: Path, device: torch.device):
    saved = safe_torch_load(checkpoint_path)
    if saved.get('target_order') != DISEASES:
        raise ValueError('M1 checkpoint disease label order differs from the 7 RFMiD labels')
    config = saved.get('grid_config') or {}
    tag = checkpoint_path.parent.name
    arch = config.get('architecture', 'swin_t' if tag.startswith('swin_t_') else None)
    pre = config.get('preprocessing', 'cropped' if '_cropped_' in tag else None)
    aug = config.get('augmentation', 'flip_only' if '_flip_only_' in tag else None)
    lr = config.get('lr', 1e-4 if '_lr1e-04' in tag else None)
    if arch != 'swin_t' or pre != 'cropped' or aug != 'flip_only' or lr is None or not np.isclose(float(lr), 1e-4):
        raise ValueError(f'Expected GridSearchCV winner swin_t/cropped/flip_only/lr=1e-4; got {arch}/{pre}/{aug}/{lr}')
    model = build_model('swin_t', pretrained=False)
    model.load_state_dict(saved['model_state'], strict=True)
    model.to(device).eval()
    return model, saved


class SwinFeatures(nn.Module):
    """M1 disease logits AND last-layer features, no duplicated Swin pass."""
    def __init__(self, m1: nn.Module):
        super().__init__()
        if not isinstance(m1.head, nn.Linear):
            raise ValueError('Expected torchvision Swin-T with a linear classification head')
        self.feature_dim = m1.head.in_features
        self.register_buffer('head_weight', m1.head.weight.detach().clone())
        self.register_buffer('head_bias', m1.head.bias.detach().clone())
        m1.head = nn.Identity()
        self.encoder = m1
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        features = self.encoder(x)
        logits = F.linear(features, self.head_weight, self.head_bias)
        return features, logits


class QualityNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        self.net.classifier[-1] = nn.Linear(self.net.classifier[-1].in_features, 3)

    def forward(self, x):
        return self.net(x)


class QualityDataset(Dataset):
    def __init__(self, frame, train=False):
        self.paths = frame['image_path'].astype(str).tolist()
        self.labels = frame['quality'].map({q: i for i, q in enumerate(QUALITY_NAMES)}).to_numpy(dtype=np.int64)
        self.transform = image_transform('flip_only' if train else None)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as im:
            im = resize_pad(ImageOps.exif_transpose(im).convert('RGB'))
            x = self.transform(im)
        return x, int(self.labels[index])


def load_eyeq_manifest(manifest_path: Path):
    """Columns: image_path, quality_label (good/usable/reject), optional split.

    If split is present, uses train and val/validation ONLY and excludes test.
    Otherwise creates a reproducible stratified 80/20 train/validation split.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f'EyeQ quality manifest not found: {manifest_path}')
    df = pd.read_csv(manifest_path)
    needed = {'image_path', 'quality_label'}
    if not needed.issubset(df.columns):
        raise ValueError('EyeQ manifest must have image_path and quality_label columns')
    if df[list(needed)].isna().any().any():
        raise ValueError('EyeQ manifest contains empty image paths/quality labels')
    df['quality'] = df.quality_label.astype(str).str.strip().str.lower()
    if not df.quality.isin(QUALITY_NAMES).all():
        raise ValueError(f'EyeQ labels must be good, usable, reject. Found: {df.quality.unique()}')
    df['image_path'] = df.image_path.map(
        lambda p: str((Path(p) if Path(p).is_absolute() else manifest_path.parent / str(p)).resolve()))
    if df.image_path.duplicated().any():
        raise ValueError('Repeated EyeQ image paths; remove duplicates before splitting')
    if df.get('split') is not None:
        df['split'] = df['split'].astype(str).str.lower().str.strip()
        bad = ~df['split'].isin(['train', 'training', 'val', 'validation', 'test', 'testing'])
        if bad.any():
            raise ValueError('EyeQ split must be train, val/validation, or test')
        train = df[df.split.isin(['train', 'training'])].copy()
        valid = df[df.split.isin(['val', 'validation'])].copy()
        if len(train) == 0:
            raise ValueError('EyeQ manifest has no training entries')
        if len(valid) == 0:
            train, valid = train_test_split(train, test_size=0.20, stratify=train.quality, random_state=SEED)
    else:
        train, valid = train_test_split(df, test_size=0.20, stratify=df.quality, random_state=SEED)
    for title, split in [('EyeQ train', train), ('EyeQ validation', valid)]:
        if any(split.quality.value_counts().get(q, 0) < 2 for q in QUALITY_NAMES):
            raise ValueError(f'{title} has too few examples for a quality class')
        bad_paths = [p for p in split.image_path if not Path(p).is_file()]
        if bad_paths:
            raise FileNotFoundError(f'{title} contains missing files, example: {bad_paths[0]}')
    if set(train.image_path) & set(valid.image_path):
        raise RuntimeError('EyeQ train/validation overlap')
    return train.reset_index(drop=True), valid.reset_index(drop=True)


def train_quality_model(manifest: Path, out: Path, device, epochs=8, batch_size=16, lr=1e-4,
                        workers=0, fine_tune=False, force=False):
    ckpt = out / 'eyeq_quality.pt'
    train, valid = load_eyeq_manifest(manifest)
    sig = hashlib.sha256((short_path_hash(train.image_path) + short_path_hash(valid.image_path) +
                          str(train.quality.tolist()) + str(valid.quality.tolist()) +
                          f'|{epochs}|{lr}|{fine_tune}').encode()).hexdigest()
    if ckpt.is_file() and not force:
        saved = safe_torch_load(ckpt)
        if saved.get('eyeq_signature') != sig:
            raise ValueError('Existing EyeQ checkpoint was trained with different data/settings. '
                             'Use --force-quality to intentionally replace it, or keep existing settings.')
        print('[EYEQ CACHE HIT] Existing quality checkpoint reused', flush=True)
        return ckpt
    set_seed(SEED)
    net = QualityNet(pretrained=True).to(device)
    if not fine_tune:
        for param in net.net.features.parameters():
            param.requires_grad_(False)
    optim = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    counts = train.quality.value_counts()
    weights = torch.tensor([len(train) / (3 * counts[q]) for q in QUALITY_NAMES],
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    train_dl = DataLoader(QualityDataset(train, train=True), batch_size=batch_size,
                          shuffle=True, num_workers=workers, pin_memory=device.type=='cuda')
    valid_dl = DataLoader(QualityDataset(valid), batch_size=batch_size,
                          num_workers=workers, pin_memory=device.type=='cuda')
    best_score = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        net.train()
        # Frozen backbone: keep its batch-normalization running statistics fixed.
        if not fine_tune:
            net.net.features.eval()
        total, n = 0., 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            loss = criterion(net(x), y)
            loss.backward()
            optim.step()
            total += loss.item() * len(y)
            n += len(y)
        net.eval()
        truth, preds = [], []
        with torch.inference_mode():
            for x, y in valid_dl:
                p = net(x.to(device)).argmax(1).cpu().numpy()
                truth.extend(y.numpy().tolist())
                preds.extend(p.tolist())
        f1 = f1_score(truth, preds, average='macro', labels=[0, 1, 2], zero_division=0)
        bac = balanced_accuracy_score(truth, preds)
        history.append({'epoch': epoch, 'train_loss': total/n, 'quality_macro_f1':float(f1),
                        'quality_balanced_accuracy':float(bac)})
        print(f'EyeQ epoch {epoch}/{epochs}: loss {total/n:.4f}, val macro F1 {f1:.4f}', flush=True)
        if f1 > best_score:
            best_score = f1
            torch_save_atomic({'model_state': net.cpu().state_dict(), 'class_order': QUALITY_NAMES,
                               'eyeq_signature': sig, 'epochs': epochs,
                               'fine_tune': bool(fine_tune), 'best_validation_macro_f1': float(best_score)}, ckpt)
            net.to(device)
    pd.DataFrame(history).to_csv(out / 'eyeq_history.csv', index=False)
    saved = safe_torch_load(ckpt)
    net = QualityNet(pretrained=False).to(device).eval()
    net.load_state_dict(saved['model_state'], strict=True)
    truth, preds = [], []
    with torch.inference_mode():
        for x, y in valid_dl:
            preds.extend(net(x.to(device)).argmax(1).cpu().tolist())
            truth.extend(y.tolist())
    quality_metrics = {'macro_f1': float(f1_score(truth, preds, average='macro', labels=[0,1,2], zero_division=0)),
                       'balanced_accuracy': float(balanced_accuracy_score(truth, preds)),
                       'confusion_matrix': confusion_matrix(truth, preds, labels=[0,1,2]).tolist(),
                       'class_order': QUALITY_NAMES,
                       'n_val': len(truth), 'eyeq_validation_only': True}
    save_json(quality_metrics, out / 'eyeq_quality_validation_metrics.json')
    return ckpt


def load_quality(path, device):
    saved = safe_torch_load(path)
    if saved.get('class_order') != QUALITY_NAMES:
        raise ValueError('EyeQ checkpoint class order not good/usable/reject')
    model = QualityNet(pretrained=False)
    model.load_state_dict(saved['model_state'], strict=True)
    model.to(device).eval()
    return model


class FusionHead(nn.Module):
    """M2: M1 logits + trainable correction conditioned on EyeQ probabilities.

    The 768-D Swin encoder and the EyeQ model are frozen; this module alone is
    fitted on RFMiD TRAIN.  M3 uses Monte Carlo dropout in the fusion head.
    """
    def __init__(self, feature_dim=768, hidden=128, dropout=0.25):
        super().__init__()
        self.fc = nn.Linear(feature_dim + 3, hidden)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, len(DISEASES))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.feature_dim = feature_dim

    def forward(self, features, m1_logits, quality_probs):
        residual = self.out(self.dropout(self.act(self.fc(torch.cat([features, quality_probs], dim=1)))))
        return m1_logits + residual


class ImagePairDataset(Dataset):
    """The disease branch sees cropped input; quality branch sees padded original."""
    def __init__(self, pair_paths):
        self.items = np.asarray(pair_paths, dtype=str)
        if self.items.ndim != 2 or self.items.shape[1] != 2:
            raise ValueError('Expected (N,2) paths: padded then cropped')
        self.transform = image_transform()
        for path in self.items.reshape(-1):
            if not Path(path).is_file():
                raise FileNotFoundError(path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        padded, cropped = self.items[index]
        with Image.open(padded) as image:
            quality_x = self.transform(image.convert('RGB'))
        with Image.open(cropped) as image:
            m1_x = self.transform(image.convert('RGB'))
        return m1_x, quality_x


def cached_features(name, pair_paths, m1net, qualitynet, out, device, batch_size, workers,
                    m1_hash, quality_hash, source_fingerprint, force=False):
    """Saves frozen-model activations for fast local M2 head training."""
    out.mkdir(parents=True, exist_ok=True)
    file = out / f'feature_cache_{name}.npz'
    sig = hashlib.sha256(json.dumps({
        'paths_hash': short_path_hash(np.asarray(pair_paths).reshape(-1)),
        'm1': m1_hash, 'eyeq': quality_hash, 'preprocess': source_fingerprint,
        'branch_inputs': 'm1_cropped__quality_padded',
    }, sort_keys=True).encode()).hexdigest()
    if file.is_file() and not force:
        with np.load(file, allow_pickle=False) as obj:
            if obj['signature'].item() == sig and len(obj['features']) == len(pair_paths):
                print(f'[FEATURE CACHE HIT] {name}: {len(pair_paths)} images', flush=True)
                return {key: obj[key].copy() for key in ('features','m1_logits','quality_probs')}
    dataset = ImagePairDataset(pair_paths)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                        pin_memory=device.type=='cuda')
    features, logits, qualities = [], [], []
    m1net.eval()
    qualitynet.eval()
    with torch.inference_mode():
        for step, (cropped, padded) in enumerate(loader, 1):
            f, z = m1net(cropped.to(device))
            q = F.softmax(qualitynet(padded.to(device)), dim=1)
            features.append(f.float().cpu().numpy())
            logits.append(z.float().cpu().numpy())
            qualities.append(q.float().cpu().numpy())
            if step % 20 == 0:
                print(f'[FEATURE EXTRACT] {name}: {min(step*batch_size,len(dataset))}/{len(dataset)}', flush=True)
    record = {'features': np.concatenate(features), 'm1_logits': np.concatenate(logits),
              'quality_probs': np.concatenate(qualities)}
    tmp = file.with_name(file.name + '.tmp.npz')
    np.savez_compressed(tmp, signature=np.array(sig), **record)
    tmp.replace(file)
    print(f'[FEATURE CACHE SAVED] {name}: {len(dataset)} images', flush=True)
    return record


def macro_ap(y, probs):
    values = []
    for i, d in enumerate(DISEASES):
        if np.unique(y[:, i]).size != 2:
            raise ValueError(f'Cannot calculate macro AP: {d} has only one label class')
        values.append(average_precision_score(y[:, i], probs[:, i]))
    return float(np.mean(values))


def train_m2(train, select, y_train, y_select, out, device, m1_hash, quality_hash,
             fingerprint, epochs=12, batch_size=128, lr=1e-3, force=False,
             no_quality_ablation=False):
    checkpoint = out / ('m2_no_quality_ablation.pt' if no_quality_ablation else 'm2_quality_fusion.pt')
    sig = hashlib.sha256(json.dumps({'m1':m1_hash, 'quality':quality_hash,
                                     'source':fingerprint,'epochs':epochs, 'batch_size':batch_size,
                                     'lr':lr, 'seed':SEED,'no_quality_ablation':no_quality_ablation}, sort_keys=True).encode()).hexdigest()
    if checkpoint.is_file() and not force:
        saved = safe_torch_load(checkpoint)
        if saved.get('signature') != sig:
            raise ValueError('M2 checkpoint settings/data changed. Use --force-m2 to replace intentionally.')
        print('[M2 CACHE HIT] Reusing quality-fusion checkpoint', flush=True)
        return checkpoint
    set_seed(SEED)
    feat_dim = int(train['features'].shape[1])
    head = FusionHead(feature_dim=feat_dim).to(device)
    train_q = np.zeros_like(train['quality_probs']) if no_quality_ablation else train['quality_probs']
    sel_q = np.zeros_like(select['quality_probs']) if no_quality_ablation else select['quality_probs']
    train_ds = TensorDataset(*(torch.from_numpy(np.asarray(arr, dtype=np.float32)) for arr in
                               (train['features'],train['m1_logits'],train_q,y_train)))
    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                    generator=torch.Generator().manual_seed(SEED), num_workers=0)
    class_pos = y_train.sum(axis=0)
    class_neg = len(y_train) - class_pos
    if (class_pos == 0).any() or (class_neg == 0).any():
        raise ValueError('M2 train requires positive and negative examples for each label')
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(class_neg/class_pos, device=device, dtype=torch.float32))
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    sel_inputs = [torch.tensor(arr, dtype=torch.float32, device=device)
                  for arr in (select['features'],select['m1_logits'],sel_q)]
    best = -math.inf
    history = []
    for epoch in range(1,epochs+1):
        head.train()
        total, n = 0., 0
        for f, z, q, y in dl:
            f,z,q,y = (x.to(device) for x in (f,z,q,y))
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(head(f,z,q), y)
            loss.backward()
            opt.step()
            total += loss.item()*len(y)
            n += len(y)
        head.eval()
        with torch.inference_mode():
            probs = torch.sigmoid(head(*sel_inputs)).cpu().numpy()
        score = macro_ap(y_select, probs)
        history.append({'epoch':epoch, 'train_loss':total/n, 'selection_macro_ap':score})
        print(f'M2 epoch {epoch}/{epochs}: loss={total/n:.4f}, selection macro AP={score:.5f}', flush=True)
        if score > best:
            best = score
            torch_save_atomic({'model_state':head.cpu().state_dict(), 'feature_dim':feat_dim,
                               'target_order':list(DISEASES), 'quality_classes':list(QUALITY_NAMES),
                               'signature':sig,'m1_sha256':m1_hash,'quality_sha256':quality_hash,
                               'epoch':epoch,'selection_macro_ap':best,
                               'no_quality_ablation':bool(no_quality_ablation)},checkpoint)
            head.to(device)
    pd.DataFrame(history).to_csv(out / 'm2_training_history.csv',index=False)
    return checkpoint


def load_head(checkpoint, device, m1_hash, quality_hash):
    saved = safe_torch_load(checkpoint)
    if saved.get('target_order') != DISEASES or saved.get('quality_classes') != QUALITY_NAMES:
        raise ValueError('M2 checkpoint labels incompatible')
    if saved.get('m1_sha256') != m1_hash or saved.get('quality_sha256') != quality_hash:
        raise ValueError('M2 checkpoint was trained against a different M1 or EyeQ checkpoint')
    head = FusionHead(feature_dim=int(saved['feature_dim'])).to(device)
    head.load_state_dict(saved['model_state'], strict=True)
    return head.eval()


def get_logits(features, head, device, mc_passes=1, seed=SEED):
    x = [torch.tensor(features[k], device=device, dtype=torch.float32) for k in
         ('features','m1_logits','quality_probs')]
    if mc_passes < 1:
        raise ValueError('mc_passes must be >= 1')
    result = []
    with torch.inference_mode():
        head.eval()
        if mc_passes > 1:
            # Activate dropout ONLY in fusion head. Swin and EyeQ remain in eval mode.
            head.dropout.train()
            torch.manual_seed(seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(seed)
        for _ in range(mc_passes):
            parts = []
            for start in range(0, len(x[0]), 256):
                end = start+256
                parts.append(head(x[0][start:end],x[1][start:end],x[2][start:end]).cpu().numpy())
            result.append(np.concatenate(parts,axis=0))
    head.eval()
    return np.stack(result,axis=0)


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1/(1+np.exp(-np.clip(x,-40,40)))


def expected_calibration_error(y, p, bins=10):
    y = np.asarray(y,dtype=int)
    p = np.asarray(p,dtype=float)
    eces = []
    for i in range(y.shape[1]):
        ece = 0.
        for j in range(bins):
            lo,hi = j/bins,(j+1)/bins
            mask = (p[:,i]>=lo)&((p[:,i]<hi) if j<bins-1 else (p[:,i]<=hi))
            if mask.any():
                ece += mask.mean()*abs(y[mask,i].mean()-p[mask,i].mean())
        eces.append(ece)
    return float(np.mean(eces))


def fit_temperatures(logits, targets):
    if logits.shape != targets.shape:
        raise ValueError('Calibration logits and targets do not match')
    log_temperature = nn.Parameter(torch.zeros(logits.shape[1], dtype=torch.float64))
    z = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(targets, dtype=torch.float64)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.2, max_iter=100, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        temp = log_temperature.exp()
        objective = F.binary_cross_entropy_with_logits(z/temp, y) + 0.01*log_temperature.square().mean()
        objective.backward()
        return objective
    optimizer.step(closure)
    return log_temperature.detach().exp().clamp(0.5,5).numpy()


def tune_thresholds(y,p):
    threshold_candidates = np.linspace(.10,.90,81)
    values = []
    for j in range(p.shape[1]):
        scores = np.array([f1_score(y[:,j], p[:,j]>=threshold, zero_division=0)
                           for threshold in threshold_candidates])
        # On ties choose threshold nearest 0.5 (and deterministic ascending candidate index).
        idx = np.flatnonzero(np.isclose(scores, scores.max()))
        values.append(float(threshold_candidates[idx[np.argmin(abs(threshold_candidates[idx]-.5))]]))
    return np.array(values)


def uncertainty_score(probs, q, mc_variance, var_scale):
    ent = -(probs*np.log(np.clip(probs,1e-7,1)) +
            (1-probs)*np.log(np.clip(1-probs,1e-7,1)))/math.log(2)
    mean_ent = ent.mean(axis=1)
    var = np.clip(mc_variance.mean(axis=1)/max(var_scale,1e-8),0,1)
    return .45*q[:,2] + .35*var + .20*mean_ent


def fit_m3(calibr, y_cal, head, device, out, m1_hash, quality_hash, m2_hash,
           mc_passes=20, target_coverage=.80, force=False):
    if not 0 < target_coverage < 1:
        raise ValueError('Target coverage must be between zero and one')
    path = out/'m3_calibration.json'
    sig = hashlib.sha256(json.dumps({'m1':m1_hash,'quality':quality_hash,'m2':m2_hash,
                                    'mc_passes':mc_passes,'coverage':target_coverage,
                                    'calibration_n':len(y_cal),'calibration_label_hash':hashlib.sha256(y_cal.astype('uint8').tobytes()).hexdigest()},
                                   sort_keys=True).encode()).hexdigest()
    if path.is_file() and not force:
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved.get('signature') != sig:
            raise ValueError('M3 calibration changed; use --force-m3 intentionally')
        print('[M3 CACHE HIT] Calibration and referral policy reused')
        return saved
    draws = get_logits(calibr,head,device,mc_passes=mc_passes)
    raw_logits = draws.mean(axis=0)
    variance = sigmoid(draws).var(axis=0)
    temp = fit_temperatures(raw_logits, y_cal)
    prob = sigmoid(raw_logits/temp)
    thresholds = tune_thresholds(y_cal,prob)
    var_scale = float(max(1e-7,np.quantile(variance.mean(axis=1),.90)))
    risk = uncertainty_score(prob,calibr['quality_probs'],variance,var_scale)
    score_threshold = float(np.quantile(risk, target_coverage))
    results = {'signature':sig, 'class_order':list(DISEASES),
               'temperatures':temp.tolist(), 'thresholds':thresholds.tolist(),
               'referral_score_threshold':score_threshold, 'mc_variance_scale':var_scale,
               'target_coverage':float(target_coverage),'mc_passes':int(mc_passes),
               'm1_sha256':m1_hash, 'm2_sha256':m2_hash,'quality_sha256':quality_hash,
               'calibration_n':int(len(y_cal)),
               'raw_ece_calibration':expected_calibration_error(y_cal,sigmoid(raw_logits)),
               'temp_ece_calibration':expected_calibration_error(y_cal,prob),
               'raw_brier_calibration':float(np.mean((sigmoid(raw_logits)-y_cal)**2)),
               'temp_brier_calibration':float(np.mean((prob-y_cal)**2)),
               'note':'Parameters fitted on calibration set. These calibration metrics are not independent performance estimates.'}
    save_json(results,path)
    return results


def assert_calibration_matches(cal, m1_hash, quality_hash, m2_hash):
    if cal.get('class_order') != DISEASES:
        raise ValueError('M3 calibration class order mismatches RFMiD')
    if any(cal.get(k) != value for k,value in [('m1_sha256',m1_hash),('quality_sha256',quality_hash),('m2_sha256',m2_hash)]):
        raise ValueError('M3 calibration belongs to different model checkpoint(s)')


def scores_all(features, head, cal, device, no_quality_head=None):
    m1_probs = sigmoid(features['m1_logits'])
    m2_logits = get_logits(features,head,device,mc_passes=1)[0]
    m2_probs = sigmoid(m2_logits)
    ablation_probs = None
    if no_quality_head is not None:
        masked = {**features, 'quality_probs': np.zeros_like(features['quality_probs'])}
        ablation_probs = sigmoid(get_logits(masked,no_quality_head,device,mc_passes=1)[0])
    draws = get_logits(features,head,device,mc_passes=int(cal['mc_passes']))
    variance = sigmoid(draws).var(axis=0)
    temp = np.asarray(cal['temperatures'])
    m3_probs = sigmoid(draws.mean(axis=0)/temp)
    risk = uncertainty_score(m3_probs,features['quality_probs'],variance,float(cal['mc_variance_scale']))
    accepted = risk <= cal['referral_score_threshold']
    predictions = {'M1':m1_probs}
    if ablation_probs is not None:
        predictions['M2_no_quality'] = ablation_probs
    predictions.update(M2=m2_probs,M3=m3_probs)
    return predictions, accepted, risk, variance


def metric_rows(y,p,thresholds):
    """Binary one-vs-rest values with TN/FP/FN/TP for each of the seven labels."""
    if y.shape != p.shape:
        raise ValueError('Metric labels and prediction shapes must match')
    thresholds = np.asarray(thresholds,dtype=float)
    prediction = p >= thresholds.reshape(1,-1)
    rows = []
    for j,d in enumerate(DISEASES):
        true = y[:,j].astype(int)
        pred = prediction[:,j].astype(int)
        tn,fp,fn,tp = confusion_matrix(true,pred,labels=[0,1]).ravel()
        valid = len(np.unique(true)) == 2
        rows.append({'disease':d,'support_positive':int(true.sum()),'support_negative':int(len(true)-true.sum()),
                     'TP':int(tp),'TN':int(tn),'FP':int(fp),'FN':int(fn),
                     'threshold':float(thresholds[j]),
                     'precision':float(precision_score(true,pred,zero_division=0)),
                     'recall_sensitivity':float(recall_score(true,pred,zero_division=0)),
                     'specificity':float(tn/(tn+fp)) if (tn+fp) else np.nan,
                     'f1':float(f1_score(true,pred,zero_division=0)),
                     'average_precision':float(average_precision_score(true,p[:,j])) if valid else np.nan,
                     'auroc':float(roc_auc_score(true,p[:,j])) if valid else np.nan,
                     'brier':float(np.mean((p[:,j]-true)**2))})
    return pd.DataFrame(rows)


def aggregate_metrics(y,p,thresholds):
    rows = metric_rows(y,p,thresholds)
    pred = p >= np.asarray(thresholds).reshape(1,-1)
    tp = rows.TP.sum(); fp=rows.FP.sum(); fn=rows.FN.sum(); tn=rows.TN.sum()
    summary = {'images':len(y), 'macro_precision':float(rows.precision.mean()),
               'macro_recall_sensitivity':float(rows.recall_sensitivity.mean()),
               'macro_specificity':float(rows.specificity.mean()),
               'macro_f1':float(rows.f1.mean()),
               'micro_precision':float(tp/(tp+fp)) if tp+fp else 0.,
               'micro_recall_sensitivity':float(tp/(tp+fn)) if tp+fn else 0.,
               'micro_specificity':float(tn/(tn+fp)) if tn+fp else float('nan'),
               'micro_f1':float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,
               'macro_ap':float(rows.average_precision.mean()),
               'macro_auroc':float(rows.auroc.mean()),
               'micro_ap':float(average_precision_score(y.ravel(),p.ravel())) if len(np.unique(y))==2 else float('nan'),
               'micro_auroc':float(roc_auc_score(y.ravel(),p.ravel())) if len(np.unique(y))==2 else float('nan'),
               'exact_match_accuracy':float(np.all(pred == y,axis=1).mean()),
               'hamming_loss':float((pred != y).mean()),
               'ece':expected_calibration_error(y,p),
               'brier':float(np.mean((p-y)**2))}
    return summary,rows


def risk_coverage(y,p,thresholds,risk):
    """Error is per-image multi-label Hamming loss; lower AURC is better."""
    errors = ((p>=np.asarray(thresholds)) != y).mean(axis=1)
    indices = np.argsort(risk,kind='stable')
    csum = np.cumsum(errors[indices])
    coverage = np.arange(1,len(y)+1)/len(y)
    selective_risk = csum/np.arange(1,len(y)+1)
    # area from zero coverage, with origin risk=0; all methods use this convention
    cx, cy = np.r_[0, coverage], np.r_[0, selective_risk]
    aurc = float(np.sum((cy[1:]+cy[:-1])*.5*(cx[1:]-cx[:-1])))
    return pd.DataFrame({'coverage':coverage,'selective_hamming_error':selective_risk}),aurc


def save_eval_tables(split, y, predictions, accepted, risk, cal, out: Path, ids,
                     plot=True, bootstrap=0):
    out.mkdir(parents=True,exist_ok=True)
    summaries=[]
    rows=[]
    for model,prob in predictions.items():
        thresholds = np.asarray(cal['thresholds']) if model=='M3' else np.full(len(DISEASES),.5)
        metrics, per_class = aggregate_metrics(y,prob,thresholds)
        metrics.update(model=model, subset=split, coverage=1.0, referral_rate=0.0,
                       interpretation='all images; M1/M2 threshold 0.5, M3 validation-calibrated')
        summaries.append(metrics)
        per_class.insert(0,'model',model)
        rows.append(per_class)
    if accepted.any():
        m3_retained,per_retained = aggregate_metrics(y[accepted],predictions['M3'][accepted],
                                                     np.asarray(cal['thresholds']))
        m3_retained.update(model='M3_retained',subset=split,coverage=float(accepted.mean()),
                           referral_rate=float(1-accepted.mean()),interpretation='accepted images only')
        summaries.append(m3_retained)
        per_retained.insert(0,'model','M3_retained')
        rows.append(per_retained)
    else:
        raise RuntimeError('M3 referred every image: cannot report retained metrics')
    curves,aurc = risk_coverage(y,predictions['M3'],cal['thresholds'],risk)
    pd.DataFrame(summaries).assign(m3_aurc=aurc).to_csv(out/'overall_metrics.csv',index=False)
    pd.concat(rows,ignore_index=True).to_csv(out/'per_disease_metrics.csv',index=False)
    curves.to_csv(out/'m3_risk_coverage.csv',index=False)
    frame = pd.DataFrame({'ID':list(ids),'m3_uncertainty_score':risk,
                          'm3_accepted':accepted.astype(int),
                          'm3_referral_recommended':(~accepted).astype(int)})
    for model,prob in predictions.items():
        for i,d in enumerate(DISEASES):
            frame[f'{model}_{d}_probability'] = prob[:,i]
    frame.to_csv(out/'per_image_predictions.csv',index=False)
    if plot:
        plt.figure(figsize=(6,4))
        plt.plot(curves.coverage,curves.selective_hamming_error)
        plt.xlabel('Coverage (fraction of accepted images)')
        plt.ylabel('Hamming error among accepted images')
        plt.title(f'M3 risk–coverage | AURC = {aurc:.4f}')
        plt.tight_layout()
        plt.savefig(out/'m3_risk_coverage.png',dpi=180)
        plt.close()
        plt.figure(figsize=(6,4))
        for model,prob in predictions.items():
            confidence = prob.reshape(-1)
            correct = y.reshape(-1)
            bins = np.linspace(0,1,11)
            xvals,yvals = [],[]
            for j in range(10):
                mask = (confidence>=bins[j]) & (confidence<(bins[j+1]) if j<9 else confidence<=bins[j+1])
                if mask.any():
                    xvals.append(float(confidence[mask].mean()))
                    yvals.append(float(correct[mask].mean()))
            plt.plot(xvals,yvals,marker='o',label=model)
        plt.plot([0,1],[0,1],'--',color='grey',label='Perfect calibration')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Observed positive fraction')
        plt.title('Multi-label reliability diagram')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out/'reliability_diagram.png',dpi=180)
        plt.close()
    if bootstrap:
        rng=np.random.default_rng(SEED)
        ci=[]
        n=len(y)
        for model,prob in predictions.items():
            values=[]
            thresholds=np.asarray(cal['thresholds']) if model=='M3' else np.full(len(DISEASES),.5)
            for _ in range(bootstrap):
                indices=rng.choice(n,n,replace=True)
                stats,_=aggregate_metrics(y[indices],prob[indices],thresholds)
                values.append([stats['macro_ap'],stats['macro_f1'],stats['macro_specificity']])
            arr=np.asarray(values)
            for j,key in enumerate(('macro_ap','macro_f1','macro_specificity')):
                ci.append({'model':model,'metric':key,'ci_2_5':float(np.nanquantile(arr[:,j],.025)),
                           'ci_97_5':float(np.nanquantile(arr[:,j],.975))})
        pd.DataFrame(ci).to_csv(out/'bootstrap_95ci.csv',index=False)
    return pd.DataFrame(summaries)


def open_degraded(image, kind, severity, seed):
    """Deterministic, controlled synthetic distortions on source RGB pixels."""
    rng=np.random.default_rng(seed)
    rgb=np.asarray(ImageOps.exif_transpose(image).convert('RGB')).copy()
    h,w=rgb.shape[:2]
    if kind=='blur':
        result=cv2.GaussianBlur(rgb,(0,0),sigmaX=1.5*severity)
    elif kind=='motion':
        k=5 if severity==1 else 11
        kernel=np.zeros((k,k),dtype=np.float32)
        kernel[k//2,:]=1/k
        result=cv2.filter2D(rgb,-1,kernel)
    elif kind=='brightness':
        result=np.clip(rgb.astype(float)*(0.70 if severity==1 else .45),0,255).astype('uint8')
    elif kind=='contrast':
        alpha=.75 if severity==1 else .5
        result=np.clip((rgb.astype(float)-127.5)*alpha+127.5,0,255).astype('uint8')
    elif kind=='glare':
        yy,xx=np.ogrid[:h,:w]
        radius=max(h,w)*(.18 if severity==1 else .3)
        cy,cx=int(.3*h),int(.68*w)
        spot=np.exp(-((yy-cy)**2+(xx-cx)**2)/(2*(radius*.6)**2))[:,:,None]
        alpha=(.35 if severity==1 else .7)*spot
        result=np.clip(rgb*(1-alpha)+255*alpha,0,255).astype('uint8')
    elif kind=='noise':
        sigma=8 if severity==1 else 20
        result=np.clip(rgb.astype(float)+rng.normal(0,sigma,rgb.shape),0,255).astype('uint8')
    elif kind=='jpeg':
        import io
        buffer=io.BytesIO()
        Image.fromarray(rgb).save(buffer,'JPEG',quality=45 if severity==1 else 15)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')
    else:
        raise ValueError(f'Unknown degradation {kind}')
    return Image.fromarray(result)


def inference_from_images(images, m1net, qualitynet, device, batch_size=8):
    """Raw original/PIL images -> cropped Swin + original padded EyeQ predictions."""
    f,z,q=[],[],[]
    trans=image_transform()
    for start in range(0,len(images),batch_size):
        cropped_batch,padded_batch=[],[]
        for item in images[start:start+batch_size]:
            image = Image.open(item) if isinstance(item,(str,Path)) else item
            try:
                rgb=ImageOps.exif_transpose(image).convert('RGB')
                crop,_=retina_bbox_crop(rgb)
                cropped_batch.append(trans(resize_pad(crop)))
                padded_batch.append(trans(resize_pad(rgb)))
            finally:
                if isinstance(item,(str,Path)):
                    image.close()
        with torch.inference_mode():
            features,logits=m1net(torch.stack(cropped_batch).to(device))
            quality=F.softmax(qualitynet(torch.stack(padded_batch).to(device)),dim=1)
        f.append(features.cpu().numpy()); z.append(logits.cpu().numpy()); q.append(quality.cpu().numpy())
    return {'features':np.concatenate(f),'m1_logits':np.concatenate(z),'quality_probs':np.concatenate(q)}


def evaluate_degradation(frame, split, m1net, qualitynet, head, cal, device, out, severities,
                         batch_size=8, root=None, ablation_head=None):
    if root is None:
        raise ValueError('root required for source image paths')
    summary=[]
    for kind in ('blur','motion','brightness','contrast','glare','noise','jpeg'):
        for severity in severities:
            # Decode and degrade ONLY one batch at a time. Full-resolution RFMiD
            # images otherwise consume several GB when accumulated in RAM.
            chunks=[]
            for start in range(0,len(frame),batch_size):
                images=[]
                for j in range(start,min(len(frame),start+batch_size)):
                    relative=frame.iloc[j].source_image_path
                    path=Path(relative)
                    if not path.is_absolute():
                        path=root/path
                    with Image.open(path) as raw:
                        images.append(open_degraded(raw,kind,severity,SEED+j))
                chunks.append(inference_from_images(images,m1net,qualitynet,device,batch_size))
                for image in images:
                    image.close()
            feats={key:np.concatenate([c[key] for c in chunks])
                   for key in ('features','m1_logits','quality_probs')}
            preds,accept,risk,_=scores_all(feats,head,cal,device,ablation_head)
            folder=out/f'{kind}_severity{severity}'
            results=save_eval_tables(f'{split}_{kind}_severity{severity}',
                                     frame[DISEASES].to_numpy(dtype=int),preds,accept,risk,cal,
                                     folder,frame.ID.astype(str),plot=False)
            results.insert(1,'degradation',kind)
            results.insert(2,'severity',severity)
            summary.append(results)
            print(f'[ROBUSTNESS] {kind}, severity {severity}: '+
                  ', '.join(f'{r.model} AP={r.macro_ap:.4f}' for r in results.itertuples()),flush=True)
    combined=pd.concat(summary,ignore_index=True)
    combined.to_csv(out/'robustness_comparison.csv',index=False)
    return combined


def get_artifacts(root, m1_checkpoint, device):
    out=root/'revised_supervisor_experiments_v2'/MODEL_DIR_NAME
    eyeq_ckpt=out/'eyeq_quality.pt'
    m2_ckpt=out/'m2_quality_fusion.pt'
    no_quality_ckpt=out/'m2_no_quality_ablation.pt'
    m3_settings=out/'m3_calibration.json'
    for file in (eyeq_ckpt,m2_ckpt,m3_settings):
        if not file.is_file():
            raise FileNotFoundError(f'Missing {file}; run the fit command after obtaining EyeQ labels')
    m1_hash=digest_file(m1_checkpoint)
    quality_hash=digest_file(eyeq_ckpt)
    m2_hash=digest_file(m2_ckpt)
    m1,_=load_m1(m1_checkpoint,device)
    m1feat=SwinFeatures(m1).to(device).eval()
    quality=load_quality(eyeq_ckpt,device)
    head=load_head(m2_ckpt,device,m1_hash,quality_hash)
    if not no_quality_ckpt.is_file():
        raise FileNotFoundError(f'Missing no-quality ablation: {no_quality_ckpt}. Run fit.')
    no_quality_head=load_head(no_quality_ckpt,device,m1_hash,quality_hash)
    if not safe_torch_load(no_quality_ckpt).get('no_quality_ablation', False):
        raise ValueError('No-quality ablation checkpoint was not trained without quality features')
    cal=json.loads(m3_settings.read_text(encoding='utf-8'))
    assert_calibration_matches(cal,m1_hash,quality_hash,m2_hash)
    return out,m1feat,quality,head,cal,no_quality_head


def run_fit(args,root,device,m1_checkpoint):
    completed_dir=root/'revised_supervisor_experiments_v2'/MODEL_DIR_NAME
    if (completed_dir/'FINAL_TEST_STARTED.json').exists():
        raise RuntimeError('Final test has already been unlocked. Do not resume model fitting afterward.')
    data=read_manifests(root)
    out=root/'revised_supervisor_experiments_v2'/MODEL_DIR_NAME
    out.mkdir(parents=True,exist_ok=True)
    if args.quality_checkpoint:
        ckpt=Path(args.quality_checkpoint).expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        quality=load_quality(ckpt,device)
        dest=out/'eyeq_quality.pt'
        if ckpt!=dest:
            if dest.exists() and digest_file(dest)!=digest_file(ckpt):
                raise ValueError('Existing quality checkpoint differs; move/backup it before replacing')
            if not dest.exists():
                import shutil
                shutil.copy2(ckpt,dest)
        quality_ckpt=dest
    else:
        if args.eyeq_manifest is None:
            raise ValueError('M2/M3 require EyeQ quality labels. Provide --eyeq-manifest or --quality-checkpoint. '
                             'M1 alone cannot be relabeled as quality aware.')
        quality_ckpt=train_quality_model(args.eyeq_manifest.resolve(),out,device,epochs=args.quality_epochs,
                                          batch_size=args.batch_size,lr=args.quality_lr,
                                          workers=args.num_workers,fine_tune=args.quality_finetune,
                                          force=args.force_quality)
        quality=load_quality(quality_ckpt,device)
    m1_hash=digest_file(m1_checkpoint)
    quality_hash=digest_file(quality_ckpt)
    m1,_=load_m1(m1_checkpoint,device)
    m1feat=SwinFeatures(m1).to(device).eval()
    datasets={}
    for name in ('train','selection','calibration'):
        X=data['X_'+name]
        datasets[name]=cached_features(name,X,m1feat,quality,out,device,args.batch_size,
                                      args.num_workers,m1_hash,quality_hash,data['preprocess_fingerprint'],
                                      force=args.force_features)
    m2_ckpt=train_m2(datasets['train'],datasets['selection'],data['y_train'],data['y_selection'],
                     out,device,m1_hash,quality_hash,data['preprocess_fingerprint'],
                     epochs=args.m2_epochs,batch_size=args.fusion_batch_size,lr=args.m2_lr,
                     force=args.force_m2)
    no_quality_ckpt=train_m2(datasets['train'],datasets['selection'],data['y_train'],data['y_selection'],
                             out,device,m1_hash,quality_hash,data['preprocess_fingerprint'],
                             epochs=args.m2_epochs,batch_size=args.fusion_batch_size,lr=args.m2_lr,
                             force=args.force_m2,no_quality_ablation=True)
    m2_hash=digest_file(m2_ckpt)
    head=load_head(m2_ckpt,device,m1_hash,quality_hash)
    cal=fit_m3(datasets['calibration'],data['y_calibration'],head,device,out,
               m1_hash,quality_hash,m2_hash,mc_passes=args.mc_passes,
               target_coverage=args.target_coverage,force=args.force_m3)
    ablation_head=load_head(no_quality_ckpt,device,m1_hash,quality_hash)
    preds,accepted,risk,_=scores_all(datasets['selection'],head,cal,device,ablation_head)
    report=save_eval_tables('selection',data['y_selection'],preds,accepted,risk,cal,
                            out/'validation_selection',np.arange(len(data['y_selection'])))
    print('\n[SELECTION METRICS — selection data reused for model tuning; not independent test]\n',
          report[['model','macro_ap','macro_f1','macro_specificity','macro_recall_sensitivity','coverage']].to_string(index=False))
    baseline_ap=float(report.loc[report.model=='M2_no_quality','macro_ap'].iloc[0])
    quality_ap=float(report.loc[report.model=='M2','macro_ap'].iloc[0])
    print(f'[QUALITY ABLATION] M2 macro AP - M2_no_quality macro AP: {quality_ap-baseline_ap:+.5f} (selection only)')
    print('\n[M3 CALIBRATION — fitted on 192 validation images, NOT an independent test]\n',
          'ECE:',cal['raw_ece_calibration'],'->',cal['temp_ece_calibration'])
    save_json({'m1_checkpoint':str(m1_checkpoint),'m1_sha256':m1_hash,
               'eyeq_quality_checkpoint':str(quality_ckpt),'eyeq_sha256':quality_hash,
               'm2_checkpoint':str(m2_ckpt),'m2_sha256':m2_hash,
               'no_quality_checkpoint':str(no_quality_ckpt),
               'no_quality_sha256':digest_file(no_quality_ckpt),
               'm3_settings':str(out/'m3_calibration.json'),
               'm1_grid_selection_macro_ap_from_user': None,
               'test_accessed': False,'feature_cache_source_fingerprint':data['preprocess_fingerprint']},
              out/'experiment_lock.json')
    print('[FIT COMPLETE] Saved M1/M2/M3 checkpoints and validation-selection report. Official TEST not touched.')
    return out


def run_evaluate(args,root,device,m1_checkpoint):
    data=read_manifests(root)
    out,m1,quality,head,cal,ablation_head=get_artifacts(root,m1_checkpoint,device)
    f=cached_features('selection',data['X_selection'],m1,quality,out,device,args.batch_size,
                      args.num_workers,digest_file(m1_checkpoint),digest_file(out/'eyeq_quality.pt'),
                      data['preprocess_fingerprint'])
    preds,accepted,risk,_=scores_all(f,head,cal,device,ablation_head)
    report=save_eval_tables('selection',data['y_selection'],preds,accepted,risk,cal,
                            out/'validation_selection',np.arange(len(data['y_selection'])))
    print(report[['model','macro_ap','macro_f1','macro_specificity','macro_recall_sensitivity','coverage']].to_string(index=False))
    if args.robustness:
        manifest=root/'processed_datasets'/'rfmid_min100_cropped224_v2'/'val_manifest_selected.csv'
        df=pd.read_csv(manifest,dtype={'ID':'string'}).set_index('ID',drop=False)
        ids=pd.read_csv(root/'revised_supervisor_experiments_v2'/'validation_selection_manifest.csv',dtype={'ID':'string'}).ID
        selected=df.loc[ids.tolist()].reset_index(drop=True)
        evaluate_degradation(selected,'selection',m1,quality,head,cal,device,
                             out/'validation_selection_robustness',args.severities,args.batch_size,root,ablation_head)


def run_predict(args,root,device,m1_checkpoint):
    if not args.image:
        raise ValueError('predict requires --image FILE')
    path=args.image.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    # Development-time prediction must not inspect official held-out test images.
    # This checks known official RFMiD test directory names without opening them.
    for named in ('Test_Set','Testing_Set'):
        if path.is_relative_to((root/named).resolve()):
            raise ValueError('Official test images are locked during development. Use final-test only after freezing.')
    if args.m1_only:
        model,_=load_m1(m1_checkpoint,device)
        feature_model=SwinFeatures(model).to(device).eval()
        with Image.open(path) as raw:
            crop,_=retina_bbox_crop(raw)
            x=image_transform()(resize_pad(crop)).unsqueeze(0).to(device)
        with torch.inference_mode():
            _,logits=feature_model(x)
        result={'model':'M1','diseases':dict(zip(DISEASES,sigmoid(logits.cpu().numpy()[0]).tolist())),
                'calibrated':False, 'referral':'not available in M1'}
    else:
        out,m1,quality,head,cal,ablation_head=get_artifacts(root,m1_checkpoint,device)
        feats=inference_from_images([path],m1,quality,device,batch_size=1)
        pred,accepted,risk,variance=scores_all(feats,head,cal,device,ablation_head)
        result={'image':str(path),'image_quality_probabilities':dict(zip(QUALITY_NAMES,feats['quality_probs'][0].tolist())),
                'image_quality_prediction':QUALITY_NAMES[int(feats['quality_probs'][0].argmax())],
                'm1_probabilities':dict(zip(DISEASES,pred['M1'][0].tolist())),
                'm2_probabilities':dict(zip(DISEASES,pred['M2'][0].tolist())),
                'm2_no_quality_ablation_probabilities':dict(zip(DISEASES,pred['M2_no_quality'][0].tolist())),
                'm3_calibrated_probabilities':dict(zip(DISEASES,pred['M3'][0].tolist())),
                'm3_predicted_labels':[d for j,d in enumerate(DISEASES) if pred['M3'][0,j]>=cal['thresholds'][j]],
                'm3_uncertainty_score':float(risk[0]),
                'm3_mc_variance_per_disease':dict(zip(DISEASES,variance[0].tolist())),
                'referral_recommended':bool(not accepted[0]),
                'referral_reason':'quality/uncertainty score exceeds validation-calibrated policy' if not accepted[0]
                                  else 'below validation-calibrated referral threshold'}
    result['warning']='Research only, not a clinical diagnosis or patient care recommendation.'
    print(json.dumps(result,indent=2))
    if args.output:
        save_json(result,args.output)


def make_final_test_manifest(test_dir: Path, csv_path: Path):
    if not test_dir.is_dir() or not csv_path.is_file():
        raise FileNotFoundError('Provide valid official RFMiD original test image folder AND labels CSV')
    df=pd.read_csv(csv_path,dtype={'ID':'string'})
    df.columns=df.columns.str.strip()
    if not set(['ID']+DISEASES).issubset(df.columns):
        raise ValueError('Official test labels missing ID or selected disease columns')
    if df.ID.isna().any() or df.ID.duplicated().any():
        raise ValueError('Missing or repeated official test IDs')
    if df[DISEASES].isna().any().any() or not df[DISEASES].isin([0,1]).all().all():
        raise ValueError('Test labels must be 0/1')
    if len(df)!=640:
        raise ValueError(f'Expected 640 official test images; got {len(df)}')
    index={}
    for file in test_dir.rglob('*'):
        if file.is_file() and file.suffix.lower() in EXTENSIONS:
            index.setdefault(file.stem.casefold(),[]).append(file)
    paths=[]
    for ident in df.ID:
        candidates=index.get(str(ident).casefold(),[])
        if len(candidates)!=1:
            raise ValueError(f'Official test image {ident}: expected one match, found {len(candidates)}')
        paths.append(candidates[0])
    df['source_image_path']=[str(p) for p in paths]
    return df


def run_final_test(args,root,device,m1_checkpoint):
    if args.unlock_final_test != 'I_CONFIRM_FROZEN_PROTOCOL':
        raise PermissionError('Official test is locked. Supply --unlock-final-test I_CONFIRM_FROZEN_PROTOCOL '
                              'only after finalizing all choices on train/validation.')
    out,m1,quality,head,cal,ablation_head=get_artifacts(root,m1_checkpoint,device)
    finished=out/'FINAL_TEST_COMPLETED.json'
    started=out/'FINAL_TEST_STARTED.json'
    if finished.exists() or started.exists():
        raise RuntimeError('The official final-test process was already started or finished. '
                           'Do not repeatedly evaluate against the held-out test set.')
    lock_path=out/'experiment_lock.json'
    if not lock_path.is_file():
        raise RuntimeError('No frozen experiment record; run fit first')
    record=json.loads(lock_path.read_text(encoding='utf-8'))
    for name,actual in [('m1_sha256',digest_file(m1_checkpoint)),
                        ('eyeq_sha256',digest_file(out/'eyeq_quality.pt')),
                        ('m2_sha256',digest_file(out/'m2_quality_fusion.pt')),
                        ('no_quality_sha256',digest_file(out/'m2_no_quality_ablation.pt'))]:
        if record.get(name)!=actual:
            raise RuntimeError(f'{name} changed after freezing; cannot evaluate official test')
    if args.test_dir is None or args.test_csv is None:
        raise ValueError('final-test requires both --test-dir and --test-csv')
    if not args.test_dir.is_dir() or not args.test_csv.is_file():
        raise FileNotFoundError('Official test directory or official test CSV does not exist')
    test_dir=args.test_dir.resolve()
    for prohibited in (root/'Training_Set',root/'Evaluation_Set'/'Validation',
                       root/'processed_datasets',root/'revised_supervisor_experiments_v2'):
        if test_dir.is_relative_to(prohibited.resolve()):
            raise ValueError('Test image directory points to TRAIN, VALIDATION, or model outputs')
    # The audit marker is created BEFORE the first official test read. A failed
    # run cannot accidentally turn into repeated test-set peeking.
    save_json({'frozen_experiment_sha256':digest_file(lock_path),
               'test_access': 'about to read', 'do_not_tune_against_test': True}, started)
    # This is the ONLY command which opens the official test images/labels.
    test_df=make_final_test_manifest(args.test_dir.resolve(),args.test_csv.resolve())
    images=test_df.source_image_path.tolist()
    feats=inference_from_images(images,m1,quality,device,args.batch_size)
    preds,accepted,risk,_=scores_all(feats,head,cal,device,ablation_head)
    final_path=out/'final_test_clean'
    summary=save_eval_tables('official_final_test',test_df[DISEASES].to_numpy(dtype=int),
                              preds,accepted,risk,cal,final_path,test_df.ID,
                              bootstrap=args.bootstrap)
    print('[FINAL TEST CLEAN RESULTS]\n',summary[['model','macro_ap','macro_f1','macro_specificity',
                                                    'macro_recall_sensitivity','coverage']].to_string(index=False))
    if args.robustness:
        evaluate_degradation(test_df,'official_final_test',m1,quality,head,cal,device,
                             out/'final_test_robustness',args.severities,args.batch_size,root,ablation_head)
    save_json({'status':'complete','n_test':len(test_df),'frozen_experiment_sha256':digest_file(lock_path),
               'clean_metrics':str(final_path/'overall_metrics.csv'),
               'degraded_metrics':str(out/'final_test_robustness'/'robustness_comparison.csv') if args.robustness else None,
               'warning':'Test was evaluated once after freezing; do not tune against these results.'},finished)
    print('[FINAL TEST FINISHED] Results saved. Do not use them for further model selection.')


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub=parser.add_subparsers(dest='command',required=True)
    common=argparse.ArgumentParser(add_help=False)
    common.add_argument('--root',type=Path,default=default_root())
    common.add_argument('--checkpoint',type=Path,default=None,help='Your selected Swin-T cropped M1 best.pt')
    common.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    common.add_argument('--batch-size',type=int,default=8)
    common.add_argument('--num-workers',type=int,default=0)
    fit=sub.add_parser('fit',parents=[common],help='Train EyeQ M2 and calibrate M3; NO official test')
    fit.add_argument('--eyeq-manifest',type=Path,default=None)
    fit.add_argument('--quality-checkpoint',type=Path,default=None)
    fit.add_argument('--quality-epochs',type=int,default=8)
    fit.add_argument('--quality-lr',type=float,default=1e-4)
    fit.add_argument('--quality-finetune',action='store_true')
    fit.add_argument('--m2-epochs',type=int,default=12)
    fit.add_argument('--m2-lr',type=float,default=1e-3)
    fit.add_argument('--fusion-batch-size',type=int,default=128)
    fit.add_argument('--mc-passes',type=int,default=20)
    fit.add_argument('--target-coverage',type=float,default=.8)
    fit.add_argument('--force-quality',action='store_true')
    fit.add_argument('--force-m2',action='store_true')
    fit.add_argument('--force-features',action='store_true')
    fit.add_argument('--force-m3',action='store_true')
    ev=sub.add_parser('evaluate',parents=[common],help='Selection validation metrics and optional synthetic robustness')
    ev.add_argument('--robustness',action='store_true')
    ev.add_argument('--severities',nargs='+',type=int,choices=[1,2],default=[1,2])
    pre=sub.add_parser('predict',parents=[common],help='Predict an external retinal image')
    pre.add_argument('--image',type=Path,required=True)
    pre.add_argument('--output',type=Path,default=None)
    pre.add_argument('--m1-only',action='store_true')
    fin=sub.add_parser('final-test',parents=[common],help='One-time official final evaluation after freezing')
    fin.add_argument('--unlock-final-test',type=str,default=None)
    fin.add_argument('--test-dir',type=Path,default=None)
    fin.add_argument('--test-csv',type=Path,default=None)
    fin.add_argument('--robustness',action='store_true')
    fin.add_argument('--severities',nargs='+',type=int,choices=[1,2],default=[1,2])
    fin.add_argument('--bootstrap',type=int,default=200)
    args=parser.parse_args()
    root=args.root.expanduser().resolve()
    device=choose_device(args.device)
    print(f'[RFMiD] Mode={args.command}, device={device}, root={root}',flush=True)
    m1=find_m1_checkpoint(root,args.checkpoint)
    if args.command=='fit':
        run_fit(args,root,device,m1)
    elif args.command=='evaluate':
        run_evaluate(args,root,device,m1)
    elif args.command=='predict':
        run_predict(args,root,device,m1)
    elif args.command=='final-test':
        run_final_test(args,root,device,m1)


if __name__=='__main__':
    main()
