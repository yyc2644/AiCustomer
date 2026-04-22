import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

from PIL import Image, ExifTags
import numpy as np

try:
    import torch
    import open_clip
    HAS_CLIP = True
except Exception:
    HAS_CLIP = False

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# OpenMP runtime clash workaround (Windows)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')


def find_images(root: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in IMAGE_EXTS:
                files.append(os.path.join(dirpath, name))
    return files


def read_exif(img: Image.Image) -> Dict[str, str]:
    exif_data = {}
    try:
        exif = img._getexif() or {}# type: ignore
        for k, v in exif.items():
            tag = ExifTags.TAGS.get(k, k)
            exif_data[str(tag)] = str(v)
    except Exception:
        pass
    return exif_data


def is_probable_screenshot(img: Image.Image, exif: Dict[str, str]) -> bool:
    # Heuristics: common phone resolutions, no camera model, or software tag mentions
    w, h = img.size
    ratio = round(max(w, h) / max(1, min(w, h)), 2)
    common_ratios = {1.78, 1.77, 2.0, 2.06, 2.16, 2.17, 2.22, 2.33}
    software = (exif.get('Software', '') + ' ' + exif.get('ProcessingSoftware', '')).lower()
    model = (exif.get('Model', '') + ' ' + exif.get('Make', '')).lower()

    if 'screenshot' in software:
        return True

    # Typical screenshot resolutions (width or height)
    common_sides = {1080, 1170, 1179, 1125, 1242, 1284, 1440, 2160, 2340, 2400, 2532, 2556, 2688, 2778, 2796, 3200}
    if (w in common_sides or h in common_sides) and ratio in common_ratios and not model:
        return True

    return False


def edge_density(img: Image.Image) -> float:
    # simple textiness proxy: edges per pixel
    gray = img.convert('L').resize((512, 512))
    arr = np.array(gray, dtype=np.float32)
    gx = np.abs(np.diff(arr, axis=1))
    gy = np.abs(np.diff(arr, axis=0))
    edges = (gx.mean() + gy.mean()) / 255.0
    return float(edges)


def heuristic_classify(path: str) -> Tuple[str, Dict[str, float]]:
    try:
        img = Image.open(path).convert('RGB')
    except Exception:
        return 'unknown', {}

    exif = read_exif(img)
    w, h = img.size
    ratio = max(w, h) / max(1, min(w, h))

    if is_probable_screenshot(img, exif):
        return 'screenshot', {'edge_density': edge_density(img)}

    model = exif.get('Model', '')
    make = exif.get('Make', '')
    if model or make:
        return 'phone_photo', {'edge_density': edge_density(img)}

    # Heuristic: very high edge density may indicate meme/text or UI screenshot
    ed = edge_density(img)
    if ed > 0.18 and ratio > 1.4:
        return 'screenshot', {'edge_density': ed}

    if ed > 0.2:
        return 'meme_or_text', {'edge_density': ed}

    return 'other', {'edge_density': ed}


def clip_classify(paths: List[str], device: str, model_name: str, pretrained: str) -> List[Tuple[str, Dict[str, float]]]:
    # zero-shot prompts
    labels = [
        'meme image',
        'emoji image',
        'mobile phone photo',
        'mobile phone screenshot',
        'scanned document',
        'handwritten note',
        'desktop screenshot',
        'presentation slide',
    ]

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, weights_only=True
    )
    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer(model_name)
    with torch.no_grad():
        text = tokenizer(labels).to(device)
        text_features = model.encode_text(text)# type: ignore
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    results = []
    for path in paths:
        try:
            img = Image.open(path).convert('RGB')
            image = preprocess(img).unsqueeze(0).to(device)# type: ignore
            with torch.no_grad():
                image_features = model.encode_image(image)# type: ignore
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = (image_features @ text_features.T).squeeze(0)
                probs = torch.softmax(logits, dim=0).cpu().numpy()
            score_map = {labels[i]: float(probs[i]) for i in range(len(labels))}
            results.append((path, score_map))
        except Exception:
            results.append((path, {}))
    return results


def map_clip_to_category(score_map: Dict[str, float]) -> str:
    if not score_map:
        return 'unknown'
    # simple mapping by top label
    top_label = max(score_map.items(), key=lambda x: x[1])[0]
    if 'screenshot' in top_label:
        return 'screenshot'
    if 'photo' in top_label:
        return 'phone_photo'
    if 'meme' in top_label or 'emoji' in top_label:
        return 'meme_or_emoji'
    if 'document' in top_label or 'handwritten' in top_label:
        return 'document'
    if 'slide' in top_label:
        return 'slide'
    return 'other'


def main():
    parser = argparse.ArgumentParser(description='Image type classifier (heuristics + optional CLIP)')
    parser.add_argument('input_dir', help='Folder with images')
    parser.add_argument('--out-dir', default='output', help='Output folder')
    parser.add_argument('--use-clip', action='store_true', help='Use CLIP zero-shot for better accuracy')
    parser.add_argument('--device', default='cpu', help='cpu or cuda')
    parser.add_argument('--model', default='ViT-B-32', help='CLIP model name')
    parser.add_argument('--pretrained', default='laion2b_s34b_b79k', help='CLIP pretrained tag (or local safetensors)')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print('Input dir does not exist')
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    images = find_images(args.input_dir)
    if not images:
        print('No images found')
        sys.exit(0)

    # First pass: heuristics
    rows = []
    heuristic_map = {}
    for path in images:
        cat, meta = heuristic_classify(path)
        heuristic_map[path] = (cat, meta)

    clip_scores = {}
    if args.use_clip:
        if not HAS_CLIP:
            print('open_clip not available; install requirements first.')
            sys.exit(1)
        # If user passes local .safetensors path, allow it
        clip_results = clip_classify(images, args.device, args.model, args.pretrained)
        for path, score_map in clip_results:
            clip_scores[path] = score_map

    for path in images:
        h_cat, meta = heuristic_map[path]
        if args.use_clip:
            c_cat = map_clip_to_category(clip_scores.get(path, {}))
        else:
            c_cat = 'n/a'
        rows.append({
            'path': path,
            'heuristic_category': h_cat,
            'clip_category': c_cat,
            'edge_density': meta.get('edge_density', ''),
        })

    csv_path = os.path.join(args.out_dir, 'image_categories.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'heuristic_category', 'clip_category', 'edge_density'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    json_path = os.path.join(args.out_dir, 'image_categories.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f'Wrote {len(rows)} items')
    print(f'CSV : {csv_path}')
    print(f'JSON: {json_path}')


if __name__ == '__main__':
    main()
