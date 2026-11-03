#!/usr/bin/env python3
"""EyeQ good/usable/reject image-quality training for RFMiD M2/M3.

The quality checkpoint is compatible with model_deployment.QualityNet and
model_deployment.py fit --quality-checkpoint ...

Prepare uses only official EyeQ TRAIN annotations. It never reads EyeQ's
original TEST annotations or RFMiD's official TEST partition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Reuse the exact architecture & input normalization used downstream by M2/M3.
from model_deployment import (QualityNet, QUALITY_NAMES, choose_device,
                              load_quality, safe_torch_load)
from model_training import image_transform
from preprocessing import resize_pad

SEED = 42
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
CLASS_ID = {name: idx for idx, name in enumerate(QUALITY_NAMES)}
OFFICIAL_CODE_TO_LABEL = {0: 'good', 1: 'usable', 2: 'reject'}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_patient_id(name: str) -> str:
    """EyePACS naming convention: 10009_left.jpeg and 10009_right.jpeg share ID."""
    stem = Path(name).stem
    match = re.fullmatch(r'(.+?)_(left|right)', stem, flags=re.IGNORECASE)
    return match.group(1) if match else stem


def numeric_or_named_label(value) -> str:
    value = str(value).strip().lower()
    if value in CLASS_ID:
        return value
    if value in ('0', '1', '2'):
        return OFFICIAL_CODE_TO_LABEL[int(value)]
    raise ValueError(f'Unrecognized EyeQ quality annotation {value!r}. Expected 0,1,2 or good,usable,reject.')


def split_by_patient(df: pd.DataFrame, seed=SEED) -> pd.DataFrame:
    """Approx. 80/20 patient-disjoint train/validation split; never uses test."""
    if min(df.quality_label.value_counts().reindex(QUALITY_NAMES, fill_value=0)) < 5:
        raise ValueError('At least five examples per quality class are needed for 5-fold group split.')
    if df.patient_id.nunique() < 10:
        raise ValueError('Too few distinct patient IDs for patient-grouped validation split.')
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr, va = next(sgkf.split(df, y=df.quality_label.map(CLASS_ID), groups=df.patient_id))
    df = df.copy()
    df['split'] = 'train'
    df.iloc[va, df.columns.get_loc('split')] = 'val'
    if set(df.loc[df.split == 'train', 'patient_id']) & set(df.loc[df.split == 'val', 'patient_id']):
        raise RuntimeError('Patient leakage between EyeQ train and validation.')
    for split in ('train', 'val'):
        counts = df.loc[df.split == split, 'quality_label'].value_counts()
        if any(counts.get(q, 0) < 2 for q in QUALITY_NAMES):
            raise ValueError(f'{split}: an EyeQ quality class has fewer than 2 examples. {counts.to_dict()}')
    return df


def prepare_manifest(official_csv: Path, images_dir: Path, out_path: Path, seed=SEED,
                     allow_missing: bool = False) -> Path:
    """Match actual EyeQ train CSV to local EyePACS images; do not fabricate labels."""
    official_csv = official_csv.expanduser().resolve()
    images_dir = images_dir.expanduser().resolve()
    out_path = out_path.expanduser().resolve()
    if not official_csv.is_file() or not images_dir.is_dir():
        raise FileNotFoundError('Supply your real Label_EyeQ_train.csv and your EyePACS image directory.')
    if 'test' in official_csv.name.lower():
        raise ValueError('Only EyeQ train labels are allowed while developing the quality model.')
    labels = pd.read_csv(official_csv)
    if not {'image', 'quality'}.issubset(labels.columns):
        raise ValueError('Expected official EyeQ CSV columns: image, quality.')
    if labels[['image', 'quality']].isna().any().any():
        raise ValueError('EyeQ annotations contain missing filenames or quality values.')
    if labels['image'].astype(str).duplicated().any():
        raise ValueError('Duplicate EyeQ image filenames in training labels.')
    # Index image paths once. If two files match the same filename/stem, reject ambiguity.
    full_by_name: dict[str, Path] = {}
    stem_by_name: dict[str, Path] = {}
    ambiguous_stems = set()
    for candidate in images_dir.rglob('*'):
        if not candidate.is_file() or candidate.suffix.lower() not in EXTENSIONS:
            continue
        key = candidate.name.casefold()
        if key in full_by_name:
            raise ValueError(f'Ambiguous duplicate image filename in EyeQ image tree: {candidate.name}')
        full_by_name[key] = candidate
        stem = candidate.stem.casefold()
        if stem in stem_by_name:
            ambiguous_stems.add(stem)
        else:
            stem_by_name[stem] = candidate
    records, missing = [], []
    for row in labels.itertuples(index=False):
        name = Path(str(row.image)).name
        match = full_by_name.get(name.casefold())
        if match is None and Path(name).stem.casefold() not in ambiguous_stems:
            match = stem_by_name.get(Path(name).stem.casefold())
        if match is None:
            missing.append(name)
            continue
        quality = numeric_or_named_label(row.quality)
        records.append({'image_path': str(match.resolve()),
                        'quality_label': quality,
                        'patient_id': get_patient_id(name)})
    if missing and not allow_missing:
        raise FileNotFoundError(f'{len(missing)} EyeQ train images are missing. Example: {missing[:5]}. '
                                'Provide the complete images or explicitly pass --allow-missing for a partial pilot.')
    if not records:
        raise ValueError('No EyeQ images match the official quality labels.')
    frame = pd.DataFrame(records)
    if frame.image_path.duplicated().any():
        raise ValueError('Multiple annotations matched the same physical image.')
    frame = split_by_patient(frame, seed=seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    print(f'[EYEQ MANIFEST] Saved {len(frame)} real labeled images: {out_path}')
    if missing:
        print(f'[WARNING] {len(missing)} missing images excluded; partial pilot only.')
    print(pd.crosstab(frame.split, frame.quality_label).to_string())
    return out_path


def load_manifest(path: Path):
    """Read only train/val; preserve any explicit EyeQ test rows without opening them."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Missing EyeQ manifest: {path}')
    data = pd.read_csv(path)
    needed = {'image_path', 'quality_label'}
    if not needed.issubset(data.columns):
        raise ValueError(f'EyeQ manifest must contain: {sorted(needed)}')
    if data[['image_path','quality_label']].isna().any().any():
        raise ValueError('Empty EyeQ image/quality annotation.')
    data['quality_label'] = data.quality_label.map(numeric_or_named_label)
    if 'patient_id' not in data:
        data['patient_id'] = data.image_path.map(get_patient_id)
    data['patient_id'] = data.patient_id.astype(str)
    if 'split' not in data:
        data = split_by_patient(data, seed=SEED)
    else:
        data['split'] = data.split.astype(str).str.strip().str.lower().replace({'training':'train', 'validation':'val'})
        if not set(data.split).issubset({'train','val','test','testing'}):
            raise ValueError('EyeQ split must be train, val or test.')
        data = data.loc[data.split.isin(['train','val'])].copy()
        if data[data.split == 'val'].empty:
            data = split_by_patient(data, seed=SEED)
    if data.image_path.duplicated().any():
        raise ValueError('The same image occurs multiple times in EyeQ train/val.')
    if set(data.loc[data.split == 'train','patient_id']) & set(data.loc[data.split == 'val','patient_id']):
        raise ValueError('Patient overlap across EyeQ train/val. Fix manifest before training.')
    data['image_path'] = data.image_path.map(lambda p: str(
        (Path(p) if Path(p).is_absolute() else path.parent / str(p)).resolve()))
    for split in ('train', 'val'):
        subset = data.loc[data.split == split]
        counts = subset.quality_label.value_counts()
        if any(counts.get(q, 0) < 2 for q in QUALITY_NAMES):
            raise ValueError(f'{split}: too few images per quality class: {counts.to_dict()}')
        missing = [x for x in subset.image_path if not Path(x).is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} {split} images missing, e.g. {missing[0]}')
    return data.reset_index(drop=True)


class EyeQDataset(Dataset):
    """Preserve blur/illumination evidence; no cropping or color jitter."""
    def __init__(self, frame: pd.DataFrame, training: bool):
        self.paths = frame.image_path.astype(str).tolist()
        self.y = frame.quality_label.map(CLASS_ID).to_numpy(np.int64)
        self.transform = image_transform('flip_only' if training else None)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            x = self.transform(resize_pad(im))
        return x, int(self.y[i])


def loader(frame, train, batch_size, workers, device, seed=SEED):
    return DataLoader(EyeQDataset(frame, training=train), batch_size=batch_size,
                      shuffle=train, num_workers=workers, pin_memory=(device.type == 'cuda'),
                      persistent_workers=workers > 0,
                      generator=torch.Generator().manual_seed(seed))


def checkpoint_signature(manifest: Path, args) -> str:
    cfg = {k: getattr(args, k) for k in ['epochs','batch_size','lr','fine_tune','seed','no_pretrained']}
    raw = manifest.read_bytes() + json.dumps(cfg,sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def make_model(pretrained, fine_tune, device):
    try:
        net = QualityNet(pretrained=pretrained)
    except Exception as exc:
        raise RuntimeError('Could not download EfficientNet-B0 ImageNet weights. '
                           'Ensure internet access, or use --no-pretrained only for debugging.') from exc
    for param in net.net.features.parameters():
        param.requires_grad_(False)
    if fine_tune:
        for block in list(net.net.features.children())[-2:]:
            for param in block.parameters():
                param.requires_grad_(True)
    return net.to(device)


def measure(model, dl, device):
    model.eval()
    ys, preds, probabilities = [], [], []
    with torch.inference_mode():
        for x, y in dl:
            logits = model(x.to(device))
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            probabilities.append(prob)
            ys.append(y.numpy())
            preds.append(prob.argmax(axis=1))
    y = np.concatenate(ys)
    pred = np.concatenate(preds)
    prob = np.concatenate(probabilities)
    return y, pred, prob


def save_reports(y, pred, prob, frame, out):
    out.mkdir(parents=True, exist_ok=True)
    report = classification_report(y, pred, labels=[0,1,2], target_names=QUALITY_NAMES,
                                   output_dict=True, zero_division=0)
    summary = {
        'macro_f1': float(f1_score(y,pred,average='macro',labels=[0,1,2],zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y,pred)),
        'accuracy': float(accuracy_score(y,pred)),
        'n_val': int(len(y)), 'class_order': list(QUALITY_NAMES),
        'confusion_matrix': confusion_matrix(y,pred,labels=[0,1,2]).tolist(),
        'per_class': {q: {k:float(report[q][k]) for k in ('precision','recall','f1-score','support')} for q in QUALITY_NAMES},
        'eyeq_validation_only': True,
    }
    (out/'eyeq_quality_validation_metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    table = frame[['image_path','quality_label']].copy().reset_index(drop=True)
    table['predicted_quality'] = [QUALITY_NAMES[int(i)] for i in pred]
    for j, name in enumerate(QUALITY_NAMES):
        table[f'prob_{name}'] = prob[:,j]
    table.to_csv(out/'eyeq_validation_predictions.csv',index=False)
    cm = np.asarray(summary['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(5.5,5))
    ax.imshow(cm,cmap='Blues')
    ax.set_xticks(range(3),QUALITY_NAMES)
    ax.set_yticks(range(3),QUALITY_NAMES)
    ax.set_xlabel('Predicted quality')
    ax.set_ylabel('True quality')
    ax.set_title('EyeQ validation confusion matrix')
    for i in range(3):
        for j in range(3):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    fig.tight_layout()
    fig.savefig(out/'eyeq_confusion_matrix.png',dpi=170)
    plt.close(fig)
    return summary


def run_train(args):
    device = choose_device(args.device)
    seed_everything(args.seed)
    data = load_manifest(args.manifest)
    train = data.loc[data.split == 'train'].reset_index(drop=True)
    valid = data.loc[data.split == 'val'].reset_index(drop=True)
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out/'eyeq_quality.pt'
    signature = checkpoint_signature(args.manifest.expanduser().resolve(),args)
    if ckpt.is_file() and not args.force:
        old = safe_torch_load(ckpt)
        if old.get('eyeq_signature') != signature:
            raise ValueError('An EyeQ checkpoint exists with different data/settings. '
                             'Use a new --output folder or --force to intentionally retrain.')
        print(f'[CHECKPOINT REUSE] {ckpt}; skipping retraining.')
        net = load_quality(ckpt,device)
        y, pred, prob = measure(net,loader(valid,False,args.batch_size,args.workers,device),device)
        summary = save_reports(y,pred,prob,valid,out)
        print(f'[VALIDATION] macro F1={summary["macro_f1"]:.4f}, balanced accuracy={summary["balanced_accuracy"]:.4f}')
        return
    print(f'[EYEQ] Device: {device}; train={len(train)}, val={len(valid)}; classes={QUALITY_NAMES}')
    print('[TRAIN CLASS COUNTS]',train.quality_label.value_counts().to_dict())
    print('[VAL CLASS COUNTS]',valid.quality_label.value_counts().to_dict())
    print("[DEBUG] Starting model initialization...", flush=True)

    net = make_model(
        pretrained=not args.no_pretrained,
        fine_tune=args.fine_tune,
        device=device
    )

    print("[DEBUG] Model initialized successfully!", flush=True)
    tr = loader(train,True,args.batch_size,args.workers,device,seed=args.seed)
    va = loader(valid,False,args.batch_size,args.workers,device,seed=args.seed)
    print("[DEBUG] Data loaders created successfully!", flush=True)
    print(f"[DEBUG] Training batches: {len(tr)}", flush=True)
    print(f"[DEBUG] Validation batches: {len(va)}", flush=True)
    counts = train.quality_label.value_counts()
    weights = torch.tensor([len(train)/(3.0 * counts[q]) for q in QUALITY_NAMES],dtype=torch.float32,device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW((p for p in net.parameters() if p.requires_grad), lr=args.lr,weight_decay=0.01)
    best = -1.0
    patience_counter = 0
    hist = []
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    for epoch in range(1,args.epochs+1):
        net.train()
        # Hold the running statistics of frozen BatchNorm layers fixed.
        if args.fine_tune:
            for block in list(net.net.features.children())[:-2]:
                block.eval()
        else:
            net.net.features.eval()
        loss_sum, total = 0.0,0
        for batch_idx, (x, y) in enumerate(tr):
            if batch_idx % 25 == 0:
                    print(f"[TRAINING] Batch {batch_idx + 1}/{len(tr)}", flush=True)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits=net(x)
                loss=criterion(logits,y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_((p for p in net.parameters() if p.requires_grad), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum+=float(loss.item())*len(y)
            total+=len(y)
        truth,preds,_=measure(net,va,device)
        score=float(f1_score(truth,preds,average='macro',labels=[0,1,2],zero_division=0))
        balanced=float(balanced_accuracy_score(truth,preds))
        hist.append({'epoch':epoch,'train_loss':loss_sum/total,
                     'val_macro_f1':score,'val_balanced_accuracy':balanced})
        print(f'EyeQ epoch {epoch}/{args.epochs}: train loss {loss_sum/total:.4f}, '
              f'val macro F1 {score:.4f}, balanced accuracy {balanced:.4f}',flush=True)
        if score > best + 1e-6:
            best=score
            patience_counter=0
            weights_cpu = {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            payload={'model_state':weights_cpu,'class_order':list(QUALITY_NAMES),
                     'eyeq_signature':signature,'epochs':args.epochs,
                     'fine_tune':bool(args.fine_tune), 'best_validation_macro_f1':best,
                     'architecture':'efficientnet_b0', 'image_size':224,
                     'label_mapping':{str(k):v for k,v in OFFICIAL_CODE_TO_LABEL.items()},
                     'preprocessing':'resize_pad_rgb_224_imagenet_normalize',
                     'source_manifest':str(args.manifest.resolve()), 'epoch_saved':epoch}
            tmp=ckpt.with_name(ckpt.name+'.tmp')
            torch.save(payload,tmp)
            tmp.replace(ckpt)
        else:
            patience_counter+=1
        pd.DataFrame(hist).to_csv(out/'eyeq_history.csv',index=False)
        if args.patience and patience_counter >= args.patience:
            print('[EARLY STOP] No improvement in val macro F1; using best saved checkpoint.')
            break
    model=load_quality(ckpt,device)
    y,pred,prob=measure(model,va,device)
    summary=save_reports(y,pred,prob,valid,out)
    print(f'[SAVED] Compatible EyeQ quality checkpoint: {ckpt}')
    print(f'[VALIDATION] macro F1={summary["macro_f1"]:.4f}, '
          f'balanced accuracy={summary["balanced_accuracy"]:.4f}, '
          f'accuracy={summary["accuracy"]:.4f}')


def run_evaluate(args):
    device=choose_device(args.device)
    data=load_manifest(args.manifest)
    val=data[data.split == 'val'].reset_index(drop=True)
    net=load_quality(args.checkpoint.expanduser().resolve(),device)
    truth,pred,prob=measure(net,loader(val,False,args.batch_size,args.workers,device),device)
    summary=save_reports(truth,pred,prob,val,args.output.expanduser().resolve())
    print(json.dumps(summary,indent=2))


def run_predict(args):
    device=choose_device(args.device)
    image=args.image.expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    model=load_quality(args.checkpoint.expanduser().resolve(),device)
    with Image.open(image) as im:
        x=image_transform(None)(resize_pad(ImageOps.exif_transpose(im).convert('RGB'))).unsqueeze(0).to(device)
    with torch.inference_mode():
        probs = torch.softmax(model(x),dim=1).cpu().numpy()[0]
    obj={'predicted_quality':QUALITY_NAMES[int(probs.argmax())],
         'probabilities':{name:float(probs[j]) for j,name in enumerate(QUALITY_NAMES)},
         'research_only':True}
    print(json.dumps(obj,indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(obj,indent=2),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description='Train a real EyeQ quality classifier for RFMiD M2/M3')
    sub=ap.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare', help='Official EyeQ train CSV + EyePACS images -> patient-disjoint manifest')
    prep.add_argument('--labels',type=Path,required=True,help='Official Label_EyeQ_train.csv; NOT test CSV')
    prep.add_argument('--images',type=Path,required=True,help='Directory containing REAL EyePACS fundus photos')
    prep.add_argument('--output',type=Path,required=True,help='Output eyeq_manifest.csv')
    prep.add_argument('--seed',type=int,default=SEED)
    prep.add_argument('--allow-missing',action='store_true',help='Only for partial-data pilot')
    tr=sub.add_parser('train',help='Fit EfficientNet-B0 and write compatible eyeq_quality.pt')
    tr.add_argument('--manifest',type=Path,required=True)
    tr.add_argument('--output',type=Path,required=True,help='Output directory for checkpoint/results')
    tr.add_argument('--epochs',type=int,default=12)
    tr.add_argument('--batch-size',type=int,default=16)
    tr.add_argument('--lr',type=float,default=1e-4)
    tr.add_argument('--patience',type=int,default=4)
    tr.add_argument('--workers',type=int,default=0)
    tr.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    tr.add_argument('--seed',type=int,default=SEED)
    tr.add_argument('--fine-tune',action='store_true',help='Also optimize final two EfficientNet feature blocks')
    tr.add_argument('--force',action='store_true',help='Explicitly overwrite old checkpoint')
    tr.add_argument('--no-pretrained',action='store_true',help='Debug only: random initialization, not research baseline')
    ev=sub.add_parser('evaluate',help='Validation-only quality metrics (never official test)')
    ev.add_argument('--manifest',type=Path,required=True)
    ev.add_argument('--checkpoint',type=Path,required=True)
    ev.add_argument('--output',type=Path,required=True)
    ev.add_argument('--batch-size',type=int,default=16)
    ev.add_argument('--workers',type=int,default=0)
    ev.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    pred=sub.add_parser('predict',help='Run quality inference on a new image')
    pred.add_argument('--checkpoint',type=Path,required=True)
    pred.add_argument('--image',type=Path,required=True)
    pred.add_argument('--output',type=Path,default=None)
    pred.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=ap.parse_args()
    if a.command=='prepare':
        prepare_manifest(a.labels,a.images,a.output,a.seed,a.allow_missing)
    elif a.command=='train':
        run_train(a)
    elif a.command=='evaluate':
        run_evaluate(a)
    else:
        run_predict(a)


if __name__=='__main__':
    main()
