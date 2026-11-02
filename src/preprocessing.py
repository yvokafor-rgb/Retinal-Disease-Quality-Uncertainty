#!/usr/bin/env python3
"""Preprocess RFMiD training/validation images once, with portable on-disk caches.

Run from VS Code (Windows PowerShell or any terminal):
    python preprocessing.py --root "C:\\Datasets\\RFMiD"

Only official train + validation partitions are read. The test partition is not
opened. The saved CSV image paths are RELATIVE to --root so moving your project
folder to a new drive does not invalidate working paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm.auto import tqdm

DISEASES = ['DR', 'MH', 'ODC', 'TSLN', 'DN', 'MYA', 'ARMD']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
SIZE = 224
SEED = 42
CROP_RECIPE = 'retina_bbox_maxrgb12_morph5_margin002_bicubic224_black_v2'
PAD_RECIPE = 'original_exif_bicubic224_black_v1'


def default_root() -> Path:
    """Use a local RFMiD/ folder beside these scripts, or explicitly pass --root."""
    folder = Path(__file__).resolve().parent
    return folder / 'RFMiD' if (folder / 'RFMiD').is_dir() else folder


def resolve_data_path(root: Path, path_value: str) -> Path:
    """Resolve a portable path in a manifest, rejecting escapes outside dataset root."""
    value = Path(str(path_value))
    full = (value if value.is_absolute() else root / value).resolve()
    try:
        full.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f'Manifest path points outside RFMiD root: {path_value}') from exc
    return full


def relative_data_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def retina_bbox_crop(pil_image: Image.Image, threshold=12, margin=0.02):
    """Remove black fundus borders using largest bright connected region."""
    image = ImageOps.exif_transpose(pil_image).convert('RGB')
    pixels = np.asarray(image)
    h, w = pixels.shape[:2]
    mask = (pixels.max(axis=2) > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    meta = {'original_width': w, 'original_height': h, 'fallback': False}
    if count <= 1:
        meta['fallback'] = True
        return image, meta
    region = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, bw, bh, area = map(int, stats[region])
    if bw < .35 * w or bh < .35 * h or area < .08 * w * h:
        meta['fallback'] = True
        return image, meta
    mx, my = round(bw * margin), round(bh * margin)
    left, top = max(0, x - mx), max(0, y - my)
    right, bottom = min(w, x + bw + mx), min(h, y + bh + my)
    meta.update(left=left, top=top, right=right, bottom=bottom,
                crop_width=right-left, crop_height=bottom-top,
                area_before=round(area / (w*h), 4),
                area_after=round(area / ((right-left)*(bottom-top)), 4))
    return image.crop((left, top, right, bottom)), meta


def resize_pad(im: Image.Image, size=SIZE):
    return ImageOps.pad(im, (size, size), method=Image.Resampling.BICUBIC,
                        color=(0, 0, 0), centering=(.5, .5))


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
    temp.replace(path)


def load_source_manifest(split, root: Path, project: Path, force_index=False):
    """Load/index original filenames while ensuring label/path/ID agreement."""
    if split == 'train':
        directory = root / 'Training_Set' / 'Training'
        csv = root / 'Training_Set' / 'RFMiD_Training_Labels.csv'
    elif split == 'val':
        directory = root / 'Evaluation_Set' / 'Validation'
        csv = root / 'Evaluation_Set' / 'RFMiD_Validation_Labels.csv'
    else:
        raise ValueError('This script may only access train and val')
    if not directory.is_dir() or not csv.is_file():
        raise FileNotFoundError(f'Missing original folder or labels: {directory} ; {csv}')
    labels = pd.read_csv(csv, dtype={'ID': 'string'})
    labels.columns = labels.columns.str.strip()
    required = ['ID'] + DISEASES
    if not set(required).issubset(labels.columns):
        raise ValueError(f'Missing columns in {csv}; need {required}')
    labels['ID'] = labels['ID'].str.strip()
    if labels.ID.isna().any() or labels.ID.duplicated().any():
        raise ValueError(f'Duplicate or missing IDs in {split} labels')
    for name in DISEASES:
        labels[name] = pd.to_numeric(labels[name], errors='raise')
    if labels[DISEASES].isna().any().any() or not labels[DISEASES].isin([0, 1]).all().all():
        raise ValueError('Disease labels must be 0/1 with no missing values')
    cache = project / 'source_manifests' / f'{split}_original_manifest.csv'
    cols = ['ID', 'split', 'source_image_path'] + DISEASES
    if cache.is_file() and not force_index:
        try:
            old = pd.read_csv(cache, dtype={'ID': 'string'})
            valid = (set(cols).issubset(old.columns) and len(old) == len(labels)
                     and old.ID.tolist() == labels.ID.tolist()
                     and np.array_equal(old[DISEASES].to_numpy(dtype=int),
                                        labels[DISEASES].to_numpy(dtype=int))
                     and (old.split == split).all() and old.source_image_path.notna().all())
            if valid:
                # Old Colab exports may contain absolute /content/drive/... paths.
                # If any cached path is missing/outside this local root, re-index.
                rewritten = []
                for image_path in old.source_image_path:
                    resolved = resolve_data_path(root, image_path)
                    if not resolved.is_file() or not resolved.is_relative_to(directory.resolve()):
                        valid = False
                        break
                    rewritten.append(relative_data_path(root, resolved))
                if valid:
                    old['source_image_path'] = rewritten
                    old[cols].to_csv(cache, index=False)
                    print(f'[SOURCE CACHE HIT] {split}: {len(old)} original records')
                    return old[cols].reset_index(drop=True)
        except (ValueError, OSError, KeyError, TypeError):
            pass  # Safely rebuild invalid or relocated legacy path references.
    index = {}
    for path in directory.rglob('*'):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            index.setdefault(path.stem.casefold(), []).append(relative_data_path(root, path))
    paths = []
    for ident in labels.ID:
        matches = index.get(str(ident).casefold(), [])
        if len(matches) != 1:
            raise ValueError(f'{split} ID {ident}: expected exactly one original photo, found {len(matches)}')
        paths.append(matches[0])
    result = labels[required].copy()
    result.insert(1, 'split', split)
    result.insert(2, 'source_image_path', paths)
    cache.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(cache, index=False)
    print(f'[SOURCE INDEX SAVED] {cache}')
    return result[cols].reset_index(drop=True)


def source_signature(df, root: Path):
    """Hash source identifiers, labels, and file metadata without decoding pixels."""
    digest = hashlib.sha256()
    for row in df[['ID', 'split', 'source_image_path'] + DISEASES].itertuples(index=False, name=None):
        digest.update(json.dumps([str(v) for v in row], separators=(',', ':')).encode())
        stat = resolve_data_path(root, str(row[2])).stat()
        digest.update(f'\0{stat.st_size}:{stat.st_mtime_ns}'.encode())
    return digest.hexdigest()


def export_split(source, split, base_dir, source_sig, recipe, root: Path, crop=False, force=False):
    """Skip valid existing exports; regenerate when source/recipe change."""
    base_dir = Path(base_dir)
    img_dir = base_dir / 'images' / split
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest = base_dir / f'{split}_manifest_selected.csv'
    info_path = base_dir / f'{split}_cache_info.json'
    expected_files = [img_dir / f'{i:06d}.png' for i in range(len(source))]
    expected_paths = [relative_data_path(root, p) for p in expected_files]
    cached = None
    compatible = False
    if not force and manifest.is_file():
        try:
            old = pd.read_csv(manifest, dtype={'ID': 'string'})
            columns = ['ID', 'split', 'source_image_path', 'image_path'] + DISEASES
            compatible = (set(columns).issubset(old.columns) and len(old) == len(source)
                 and old.ID.tolist() == source.ID.tolist()
                 and old.source_image_path.tolist() == source.source_image_path.tolist()
                 and old.image_path.tolist() == expected_paths
                 and (old.split == split).all()
                 and np.array_equal(old[DISEASES].to_numpy(dtype=int), source[DISEASES].to_numpy(dtype=int)))
            compatible = compatible and info_path.is_file()
            if compatible:
                info = json.loads(info_path.read_text(encoding='utf-8'))
                compatible = (info.get('recipe') == recipe
                              and info.get('source_signature') == source_sig
                              and info.get('image_count') == len(source)
                              and info.get('size') == SIZE)
            if compatible:
                cached = old
        except (ValueError, KeyError, OSError, json.JSONDecodeError):
            compatible = False
    inventory = {p.name: p.stat().st_size for p in img_dir.glob('*.png') if p.is_file()}
    missing = [p for p in expected_files if inventory.get(p.name, 0) <= 0]
    if compatible and not missing:
        # Spot-check image headers to catch stale/corrupt cache without opening originals.
        try:
            for i in sorted({0, len(expected_paths)//2, len(expected_paths)-1}):
                with Image.open(expected_files[i]) as im:
                    if im.size != (SIZE, SIZE):
                        compatible = False
        except (OSError, IndexError):
            compatible = False
    if compatible and not missing:
        print(f'[CACHE HIT] {"cropped" if crop else "padded"} {split}: {len(cached)} images reused, no new preprocessing')
        return cached

    reexport = force or not compatible
    if reexport:
        print(f'[CACHE REBUILD] {"cropped" if crop else "padded"} {split}: missing or incompatible cache')
    else:
        print(f'[CACHE REPAIR] {"cropped" if crop else "padded"} {split}: {len(missing)} missing files')
    saved_by_id = cached.set_index('ID') if compatible else None
    rows = []
    updated = 0
    missing_names = {Path(p).name for p in missing}
    for i, row in tqdm(source.reset_index(drop=True).iterrows(), total=len(source),
                       desc=f'Export {"crop" if crop else "pad"} {split}'):
        path = expected_files[i]
        needs_update = reexport or path.name in missing_names
        metadata = {}
        if needs_update:
            with Image.open(resolve_data_path(root, row.source_image_path)) as original:
                if crop:
                    roi, metadata = retina_bbox_crop(original)
                else:
                    roi = ImageOps.exif_transpose(original).convert('RGB')
                resize_pad(roi).save(path, 'PNG')
            updated += 1
        elif crop and saved_by_id is not None:
            previous = saved_by_id.loc[str(row.ID)]
            keys = ['original_width', 'original_height', 'fallback', 'left', 'top', 'right',
                    'bottom', 'crop_width', 'crop_height', 'area_before', 'area_after']
            metadata = {key: previous[key] for key in keys if key in previous.index}
        record = {'ID': row.ID, 'split': split, 'source_image_path': row.source_image_path,
                  'image_path': expected_paths[i], **{d: int(row[d]) for d in DISEASES}, **metadata}
        rows.append(record)
    df = pd.DataFrame(rows)
    tmp = manifest.with_suffix('.tmp.csv')
    df.to_csv(tmp, index=False)
    tmp.replace(manifest)
    save_json({'recipe': recipe, 'source_signature': source_sig,
               'image_count': len(source), 'size': SIZE, 'test_set': 'never accessed'}, info_path)
    print(f'[EXPORT SAVED] {split}: wrote {updated}, reused {len(df) - updated}; manifest={manifest}')
    return df


def exact_train_validation_audit(train: pd.DataFrame, val: pd.DataFrame,
                                 project_dir: Path, train_sig: str, val_sig: str, root: Path) -> None:
    """Check decoded RGB-pixel duplicates once per source signature; never inspect test.

    Exact hashes will not catch recompressed or resized near-duplicates; separately
    inspect the pHash candidate pairs reported by the original notebook.
    """
    report = project_dir / 'exact_train_val_pixel_hash_audit.json'
    signature = hashlib.sha256((train_sig + val_sig).encode()).hexdigest()
    if report.is_file():
        try:
            cached = json.loads(report.read_text(encoding='utf-8'))
            if cached.get('signature') == signature and cached.get('passed') is True:
                print('[AUDIT CACHE HIT] Exact train/validation audit previously passed')
                return
        except (ValueError, OSError):
            pass

    def pixel_hash(p):
        with Image.open(p) as image:
            pixels = np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
            digest = hashlib.sha256(str(pixels.shape).encode('ascii'))
            digest.update(pixels.tobytes())
            return digest.hexdigest()

    training = {pixel_hash(resolve_data_path(root, p)) for p in tqdm(train.source_image_path, desc='Train exact hashes')}
    validation = {pixel_hash(resolve_data_path(root, p)) for p in tqdm(val.source_image_path, desc='Val exact hashes')}
    overlaps = training.intersection(validation)
    if overlaps:
        raise RuntimeError(f'Found {len(overlaps)} identical train/validation pixel hashes. '
                           'Investigate image leakage before model training.')
    save_json({'signature': signature, 'passed': True,
               'train_unique_hashes': len(training), 'val_unique_hashes': len(validation)}, report)
    print('[AUDIT SAVED] No exact decoded-pixel train/validation overlap')

def split_validation(validation, project_dir, seed=SEED):
    """Persist one multi-label-stratified selection/calibration split by image ID."""
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    select_file = project_dir / 'validation_selection_manifest.csv'
    calibr_file = project_dir / 'validation_calibration_manifest.csv'
    split_info = project_dir / 'validation_split_info.json'
    labels_digest = hashlib.sha256()
    labels_digest.update(json.dumps(validation.ID.astype(str).tolist()).encode('utf-8'))
    labels_digest.update(validation[DISEASES].to_numpy(dtype=np.uint8).tobytes())
    label_signature = labels_digest.hexdigest()
    subsets = None
    if select_file.is_file() and calibr_file.is_file():
        try:
            old_select = pd.read_csv(select_file, dtype={'ID': 'string'})
            old_calibr = pd.read_csv(calibr_file, dtype={'ID': 'string'})
            id_set = set(validation.ID)
            all_saved = list(old_select.ID) + list(old_calibr.ID)
            provenance_ok = False
            if split_info.is_file():
                info = json.loads(split_info.read_text(encoding='utf-8'))
                provenance_ok = (info.get('labels_signature') == label_signature and
                                 info.get('seed') == seed)
            else:
                # For older manifests, only adopt when every saved disease label
                # matches the CURRENT official validation labels for that ID.
                indexed = validation.set_index('ID')
                provenance_ok = all(
                    set(df.ID).issubset(set(indexed.index)) and
                    np.array_equal(indexed.loc[df.ID.tolist(), DISEASES].to_numpy(dtype=int),
                                   df[DISEASES].to_numpy(dtype=int))
                    for df in (old_select, old_calibr)
                )
            if (provenance_ok and len(all_saved) == len(validation)
                    and len(set(all_saved)) == len(all_saved) and set(all_saved) == id_set):
                indexed = validation.set_index('ID', drop=False)
                subsets = (indexed.loc[old_select.ID.tolist()].reset_index(drop=True),
                           indexed.loc[old_calibr.ID.tolist()].reset_index(drop=True))
                print('[VALIDATION SPLIT REUSED] Same selection/calibration image IDs')
        except (OSError, ValueError, KeyError):
            pass
    if subsets is None:
        stratifier = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=0.30, random_state=seed)
        selection_idx, calibration_idx = next(
            stratifier.split(validation, validation[DISEASES].to_numpy(dtype=int)))
        subsets = (validation.iloc[selection_idx].reset_index(drop=True),
                   validation.iloc[calibration_idx].reset_index(drop=True))
    selection, calibration = subsets
    if set(selection.ID) & set(calibration.ID) or len(selection) + len(calibration) != len(validation):
        raise RuntimeError('Invalid validation split: overlap or missing rows')
    for name, df in [('selection', selection), ('calibration', calibration)]:
        positives = df[DISEASES].sum()
        if ((positives < 2) | ((len(df) - positives) < 2)).any():
            raise ValueError(f'Too few positives/negatives in {name}: {positives.to_dict()}')
        df.to_csv(project_dir / f'validation_{name}_manifest.csv', index=False)
    save_json({'labels_signature': label_signature, 'seed': seed,
               'selection_ids': len(selection), 'calibration_ids': len(calibration)}, split_info)
    print(f'[VALIDATION] {len(selection)} selection, {len(calibration)} calibration')
    return selection, calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=default_root(),
                        help='Local folder containing Training_Set and Evaluation_Set')
    parser.add_argument('--force', action='store_true', help='Regenerate ALL cropped and padded PNGs')
    parser.add_argument('--force-index', action='store_true', help='Re-index original image names')
    parser.add_argument('--skip-exact-audit', action='store_true',
                        help='Skip train/val exact-pixel audit (not recommended for final research)')
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not (root / 'Training_Set').is_dir():
        raise FileNotFoundError(f'No Training_Set under {root}. Set --root to your RFMiD dataset folder.')
    print(f'[LOCAL DATA ROOT] {root}')
    project = root / 'revised_supervisor_experiments_v2'
    project.mkdir(parents=True, exist_ok=True)
    train = load_source_manifest('train', root, project, force_index=args.force_index)
    val = load_source_manifest('val', root, project, force_index=args.force_index)
    if len(train) != 1920 or len(val) != 640:
        raise ValueError(f'Unexpected official RFMiD sizes: train={len(train)}, val={len(val)}')
    if not (train[DISEASES].sum() >= 100).all():
        raise ValueError('Selected training disease columns do not meet the >=100 positive rule')
    if set(train.source_image_path) & set(val.source_image_path):
        raise RuntimeError('Train/validation source path overlap')
    cropped = root / 'processed_datasets' / 'rfmid_min100_cropped224_v2'
    padded = project / 'padded224_grid_control'
    sig_train, sig_val = source_signature(train, root), source_signature(val, root)
    if not args.skip_exact_audit:
        exact_train_validation_audit(train, val, project, sig_train, sig_val, root)
    for split, frame in [('train', train), ('val', val)]:
        signature = sig_train if split == 'train' else sig_val
        export_split(frame, split, cropped, signature, CROP_RECIPE, root=root, crop=True, force=args.force)
        export_split(frame, split, padded, signature, PAD_RECIPE, root=root, crop=False, force=args.force)
    pd.DataFrame({'disease': DISEASES}).to_csv(cropped / 'selected_disease_columns.csv', index=False)
    cropped_val = pd.read_csv(cropped / 'val_manifest_selected.csv', dtype={'ID':'string'})
    split_validation(cropped_val, project)
    print('Finished. Model training should load cached CSV image_path entries; no test data accessed.')


if __name__ == '__main__':
    main()
