import argparse
import csv
import json
import os
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = '0'
# Windows OpenMP runtime clash workaround (PyTorch + NumPy/FAISS)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    import torch
except Exception as e:
    print('PyTorch is required. Please install dependencies first.')
    raise

try:
    import open_clip
except Exception as e:
    print('open_clip_torch is required. Please install dependencies first.')
    raise

try:
    from PIL import Image
except Exception as e:
    print('Pillow is required. Please install dependencies first.')
    raise

try:
    import cv2
except Exception:
    cv2 = None

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.wmv'}


@dataclass
class MediaItem:
    path: str
    kind: str  # image or video
    embedding: np.ndarray


def find_media_files(root: str) -> Tuple[List[str], List[str]]:
    images = []
    videos = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            full = os.path.join(dirpath, name)
            if ext in IMAGE_EXTS:
                images.append(full)
            elif ext in VIDEO_EXTS:
                videos.append(full)
    return images, videos


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert('RGB')
    return img


def sample_video_frames(path: str, frame_step_sec: float, max_frames: int) -> List[np.ndarray]:
    if cv2 is None:
        raise RuntimeError('opencv-python is required for video support')
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    step = max(int(frame_step_sec * fps), 1)
    frames = []
    idx = 0
    grabbed = True
    while grabbed and len(frames) < max_frames:
        grabbed, frame = cap.read()
        if not grabbed:
            break
        if idx % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        idx += 1
    cap.release()
    return frames


def embed_images(model, preprocess, device, images: List[Image.Image]) -> np.ndarray:
    with torch.no_grad():
        batch = torch.stack([preprocess(img) for img in images]).to(device)
        emb = model.encode_image(batch)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()


def embed_frames(model, preprocess, device, frames: List[np.ndarray]) -> np.ndarray:
    pil_images = [Image.fromarray(f) for f in frames]
    return embed_images(model, preprocess, device, pil_images)


def aggregate_embeddings(embs: np.ndarray) -> np.ndarray:
    if embs.shape[0] == 0:
        return None# type: ignore
    mean = embs.mean(axis=0)
    mean = mean / np.linalg.norm(mean)
    return mean


def build_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    if HAS_FAISS:
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))# type: ignore
        return index
    return None


def knn_search(embeddings: np.ndarray, top_k: int):
    if HAS_FAISS:
        index = build_index(embeddings)
        scores, idxs = index.search(embeddings.astype(np.float32), top_k + 1)# type: ignore
        return scores, idxs
    # brute force
    sims = embeddings @ embeddings.T
    idxs = np.argsort(-sims, axis=1)[:, : top_k + 1]# type: ignore
    scores = np.take_along_axis(sims, idxs, axis=1)
    return scores, idxs


def main():
    parser = argparse.ArgumentParser(description='AI album similarity demo')
    parser.add_argument('input_dir', help='Folder with images/videos')
    parser.add_argument('--model', default='ViT-B-32', help='CLIP model name')
    parser.add_argument('--pretrained', default='laion2b_s34b_b79k', help='CLIP pretrained tag')
    parser.add_argument('--device', default='cpu', help='cpu or cuda')
    parser.add_argument('--video-frame-step', type=float, default=2.0, help='Seconds between sampled frames')
    parser.add_argument('--max-frames-per-video', type=int, default=12, help='Max frames to sample per video')
    parser.add_argument('--top-k', type=int, default=5, help='Top similar items per item')
    parser.add_argument('--threshold', type=float, default=0.28, help='Cosine similarity threshold')
    parser.add_argument('--out-dir', default='output', help='Output folder')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print('Input dir does not exist')
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    print('Loading model...')
    pretrained = args.pretrained
    weights_only = True
    if os.path.isfile(pretrained):
        print('Using local checkpoint:', pretrained)
        if pretrained.lower().endswith('.safetensors'):
            # Safe tensor format works with weights_only=True
            weights_only = True
        else:
            # Loading a local .bin checkpoint may require weights_only=False with PyTorch 2.6+
            weights_only = False
            print('Note: loading local checkpoints can execute code if untrusted.')
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model,
        pretrained=pretrained,
        weights_only=weights_only,
    )
    model = model.to(args.device)
    model.eval()

    images, videos = find_media_files(args.input_dir)
    print(f'Found {len(images)} images and {len(videos)} videos')

    items: List[MediaItem] = []

    # Images
    for path in images:
        try:
            img = load_image(path)
            emb = embed_images(model, preprocess, args.device, [img])[0]
            items.append(MediaItem(path=path, kind='image', embedding=emb))
        except Exception as e:
            print(f'Image failed: {path} ({e})')

    # Videos
    for path in videos:
        try:
            frames = sample_video_frames(path, args.video_frame_step, args.max_frames_per_video)
            if len(frames) == 0:
                print(f'No frames from video: {path}')
                continue
            frame_embs = embed_frames(model, preprocess, args.device, frames)
            agg = aggregate_embeddings(frame_embs)
            if agg is None:
                print(f'No embeddings for video: {path}')
                continue
            items.append(MediaItem(path=path, kind='video', embedding=agg))
        except Exception as e:
            print(f'Video failed: {path} ({e})')

    if len(items) < 2:
        print('Not enough media to compare')
        sys.exit(0)

    embeddings = np.stack([it.embedding for it in items])
    scores, idxs = knn_search(embeddings, args.top_k)

    pairs = []
    for i, (row_scores, row_idxs) in enumerate(zip(scores, idxs)):
        for score, j in zip(row_scores, row_idxs):
            if i == j:
                continue
            if score < args.threshold:
                continue
            pairs.append({
                'item_a': items[i].path,
                'kind_a': items[i].kind,
                'item_b': items[j].path,
                'kind_b': items[j].kind,
                'similarity': float(score)
            })

    # Deduplicate symmetric pairs
    seen = set()
    unique_pairs = []
    for p in pairs:
        a = p['item_a']
        b = p['item_b']
        key = tuple(sorted([a, b]))
        if key in seen:
            continue
        seen.add(key)
        unique_pairs.append(p)

    json_path = os.path.join(args.out_dir, 'similar_pairs.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(unique_pairs, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(args.out_dir, 'similar_pairs.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['item_a', 'kind_a', 'item_b', 'kind_b', 'similarity'])
        writer.writeheader()
        for p in unique_pairs:
            writer.writerow(p)

    print(f'Wrote {len(unique_pairs)} similar pairs')
    print(f'JSON: {json_path}')
    print(f'CSV : {csv_path}')


if __name__ == '__main__':
    main()
