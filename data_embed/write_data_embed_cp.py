
import os
import ujson as json
import numpy as np
from PIL import Image, ImageDraw
import torch
import random
import GeodisTK
from tqdm import tqdm
import gc

# ====== Config ======
TARGET_CLASS = "ant"     # only apply face copy-paste for this class
BUILD_CP_MIRROR = True   # write augmented mirror under data_embed_pt_cp/
RNG_SEED = 123

random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

# --- 原工具函式（保留）---
def get_bounds(data, factor=1):
    min_x = min_y = max_x = max_y = 0
    abs_x = abs_y = 0
    for i in range(len(data)):
        x, y = float(data[i][0]) / factor, float(data[i][1]) / factor
        abs_x += x
        abs_y += y
        min_x, min_y = min(min_x, abs_x), min(min_y, abs_y)
        max_x, max_y = max(max_x, abs_x), max(max_y, abs_y)
    return min_x, max_x, min_y, max_y

def scale_bound(stroke, average_dimension=156):
    bounds = get_bounds(stroke)
    max_dimension = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    if max_dimension == 0:
        return np.array(stroke)
    stroke = np.array(stroke)
    stroke[:, 0:2] = stroke[:, 0:2] / (max_dimension / average_dimension)
    return stroke

def strokes_to_lines(strokes):
    strokes = scale_bound(strokes)
    x = y = 0
    lines, line, group_id = [], [], []
    cur_group_ip = -1
    for i in range(len(strokes)):
        if strokes[i][2] == 1:
            x += float(strokes[i][0])
            y += float(strokes[i][1])
            line.append([x, y])
            group_id.append(cur_group_ip)
            lines.append(line)
            line = []
        else:
            x += float(strokes[i][0])
            y += float(strokes[i][1])
            line.append([x, y])
            cur_group_ip = strokes[i][3]
    return lines, group_id

def group_lines_by_category(lines, categories):
    category_dict = {}
    for line, category in zip(lines, categories):
        category_dict.setdefault(category, []).append(line)
    return list(category_dict.values())

# ---- 連通元件與輔助 ----
def connected_components(mask_np):
    H, W = mask_np.shape
    labels = np.zeros((H, W), dtype=np.int32)
    visited = np.zeros_like(mask_np, dtype=bool)
    sizes = []
    label = 0
    for y in range(H):
        for x in range(W):
            if mask_np[y, x] and not visited[y, x]:
                label += 1
                stack = [(y, x)]
                visited[y, x] = True
                labels[y, x] = label
                sz = 1
                while stack:
                    cy, cx = stack.pop()
                    for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                        if 0 <= ny < H and 0 <= nx < W and mask_np[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            labels[ny, nx] = label
                            stack.append((ny, nx))
                            sz += 1
                sizes.append(sz)
    return labels, sizes

def bbox_of_label(labels, L):
    ys, xs = np.where(labels==L)
    if xs.size==0:
        return (0,0,labels.shape[1]-1, labels.shape[0]-1)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def largest_component_mask(mask_np):
    lab, sizes = connected_components(mask_np)
    if not sizes:
        return np.zeros_like(mask_np, dtype=np.uint8), lab, sizes, 0
    L = int(np.argmax(sizes)+1)
    return (lab==L).astype(np.uint8), lab, sizes, L

def chamfer_distance(mask_np):
    H, W = mask_np.shape
    INF = 10**7
    d = np.full((H, W), INF, np.int32)
    d[mask_np.astype(bool)] = 0
    for y in range(H):
        for x in range(W):
            v = d[y, x]
            if y>0:
                v = min(v, d[y-1,x]+3)
                if x>0: v=min(v, d[y-1,x-1]+4)
                if x<W-1: v=min(v, d[y-1,x+1]+4)
            if x>0: v=min(v, d[y,x-1]+3)
            d[y,x]=v
    for y in range(H-1,-1,-1):
        for x in range(W-1,-1,-1):
            v = d[y, x]
            if y<H-1:
                v = min(v, d[y+1,x]+3)
                if x>0: v=min(v, d[y+1,x-1]+4)
                if x<W-1: v=min(v, d[y+1,x+1]+4)
            if x<W-1: v=min(v, d[y,x+1]+3)
            d[y,x]=v
    d = d.astype(np.float32)
    if d.max()>0: d/=d.max()
    return d

# ---- 臉元件偵測（螞蟻） ----
AREA_MIN = 15
AREA_MAX = 800
HEAD_BAND_FRAC = 0.35

def detect_face_component_mask(raw01):
    """raw01: (H,W) float/uint8 in [0,1]. returns (H,W) uint8 mask or None"""
    m = (raw01 > 0.5).astype(np.uint8)
    body_mask, lab, sizes, Lmax = largest_component_mask(m)
    if Lmax == 0:
        return None
    H, W = m.shape
    x0,y0,x1,y1 = bbox_of_label(lab, Lmax)
    bw, bh = x1-x0+1, y1-y0+1

    if bw >= bh:
        band_w = max(1, int(round(bw*HEAD_BAND_FRAC)))
        head_band = np.zeros_like(m, dtype=np.uint8); head_band[y0:y1+1, x0:x0+band_w] = 1
        alt_band  = np.zeros_like(m, dtype=np.uint8); alt_band[y0:y1+1, x1-band_w+1:x1+1] = 1
    else:
        band_h = max(1, int(round(bh*HEAD_BAND_FRAC)))
        head_band = np.zeros_like(m, dtype=np.uint8); head_band[y0:y0+band_h, x0:x1+1] = 1
        alt_band  = np.zeros_like(m, dtype=np.uint8); alt_band[y1-band_h+1:y1+1, x0:x1+1] = 1

    cand = []
    for L in range(1, int(max(lab.max(), 0))+1):
        if L == Lmax: continue
        sz = sizes[L-1]
        if sz < AREA_MIN or sz > AREA_MAX: continue
        bx0,by0,bx1,by1 = bbox_of_label(lab, L)
        cx, cy = (bx0+bx1)/2, (by0+by1)/2
        if head_band[int(round(cy)), int(round(cx))] or alt_band[int(round(cy)), int(round(cx))]:
            cand.append((sz, L))
    if not cand:
        return None
    cand.sort(reverse=True)
    Lsel = cand[0][1]
    return (lab==Lsel).astype(np.uint8)

def paste_component_into_target(target_raw01, comp_mask):
    """target_raw01 (H,W) float01, comp_mask (H,W) uint8"""
    H, W = target_raw01.shape
    tgt_m = (target_raw01 > 0.5).astype(np.uint8)
    body_mask, lab, sizes, Lmax = largest_component_mask(tgt_m)
    if Lmax == 0:
        return target_raw01, None

    ys, xs = np.where(comp_mask>0)
    if xs.size==0:
        return target_raw01, None
    cx0, cy0, cx1, cy1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    bw, bh = cx1-cx0+1, cy1-cy0+1

    # valid placement map (top-left) where patch fully inside body
    from numpy.lib.stride_tricks import sliding_window_view
    area_mask = body_mask.astype(np.uint8)
    valid = np.zeros_like(area_mask, dtype=bool)
    if H >= bh and W >= bw:
        win = sliding_window_view(area_mask, (bh, bw))
        full = (win.sum(axis=(-2,-1)) == (bh*bw))
        valid[:full.shape[0], :full.shape[1]] = full
    ys2, xs2 = np.where(valid)
    if ys2.size==0:
        return target_raw01, None

    k = np.random.randint(0, ys2.size)
    dy, dx = int(ys2[k]), int(xs2[k])

    raw_aug = target_raw01.copy()
    raw_aug[dy:dy+bh, dx:dx+bw] = np.maximum(raw_aug[dy:dy+bh, dx:dx+bw],
                                             comp_mask[cy0:cy1+1, cx0:cx1+1].astype(np.float32))
    new_mask = (raw_aug > 0.5).astype(np.uint8)
    dist = chamfer_distance(new_mask)
    return raw_aug, dist

def infer_category_from_filename(filename: str):
    name = os.path.basename(filename).lower()
    if "ant" in name:
        return "ant"
    return "unknown"

# ============ 單筆處理：回傳 (raw, dist, face_mask_or_None) ============
def rasterize_groups(line_list):
    out = []
    lines, group_id = strokes_to_lines(line_list)
    grouped_lines = group_lines_by_category(lines, group_id)
    for line_for_a_group in grouped_lines:
        if len(line_for_a_group) == 1:
            continue

        img = Image.new("1", (156, 156), 0)
        draw = ImageDraw.Draw(img)
        for line in line_for_a_group:
            if len(line) < 2:
                continue
            draw.line([(int(x), int(y)) for x, y in line], fill=1, width=4)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        arr = np.array(img, dtype=np.uint8)

        arr_pad = np.zeros((256, 256), dtype=np.uint8)
        start_row = (256 - 156) // 2
        start_col = (256 - 156) // 2
        arr_pad[start_row:start_row+156, start_col:start_col+156] = arr
        arr01 = arr_pad.astype(np.float32)

        mask = arr_pad.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            geo_dist = GeodisTK.geodesic2d_raster_scan(arr_pad.astype(np.float32), mask, 0, 5)
            sd = 1 / (1 + 0.001 * np.exp(geo_dist))
            sd[np.isinf(sd)] = 0.0
            sd[np.isnan(sd)] = 0.0

        face = detect_face_component_mask(arr01)
        out.append((arr01, sd.astype(np.float32), face))
    return out

if __name__ == "__main__":
    folder_path = r"SPG\\Perceptual Grouping"
    out_base = "data_embed_pt"
    out_cp   = "data_embed_pt_cp"
    os.makedirs(out_base, exist_ok=True)
    os.makedirs(os.path.join(out_base, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_base, "valid"), exist_ok=True)
    os.makedirs(os.path.join(out_base, "test"), exist_ok=True)
    if BUILD_CP_MIRROR:
        os.makedirs(out_cp, exist_ok=True)
        os.makedirs(os.path.join(out_cp, "train"), exist_ok=True)
        os.makedirs(os.path.join(out_cp, "valid"), exist_ok=True)
        os.makedirs(os.path.join(out_cp, "test"), exist_ok=True)

    merged = []
    files = []
    print("Loading stroke files...")
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.endswith(".ndjson") or filename.endswith(".json"):
            if os.path.getsize(filepath) == 0:
                print(f"[skip empty] {filename}")
                continue
            with open(filepath, "r") as f:
                try:
                    json_data = json.load(f)
                    if "train_data" in json_data:
                        merged.extend(json_data["train_data"])
                        files.extend([filename]*len(json_data["train_data"]))
                except json.JSONDecodeError:
                    print(f"[warn] cannot parse {filename}, skipped")

    file_classes = [infer_category_from_filename(fn) for fn in files]

    # Build face library
    face_library = []
    print("Building face component library...")
    for strokes, cls_name in tqdm(zip(merged, file_classes), total=len(merged)):
        if cls_name != TARGET_CLASS: continue
        items = rasterize_groups(strokes)
        for arr01, sd, face in items:
            if face is not None:
                face_library.append(face)
    print(f"[face library] {len(face_library)} for '{TARGET_CLASS}'")

    # Estimate total
    print("Estimating total samples...")
    total_samples = 0
    per_item_groups = []
    for strokes in tqdm(merged):
        items = rasterize_groups(strokes)
        per_item_groups.append(items)
        total_samples += len(items)
    print("total:", total_samples)

    # split
    train_end = int(total_samples * 0.8)
    valid_end = train_end + int(total_samples * 0.1)
    train_count = valid_count = test_count = 0

    cursor = 0
    print("Writing...")
    for (strokes, cls_name), items in tqdm(zip(zip(merged, file_classes), per_item_groups), total=len(merged)):
        for arr01, sd, face in items:
            if cursor < train_end:
                subset = "train"; fname = f"{train_count}.pt"; train_count += 1
            elif cursor < valid_end:
                subset = "valid"; fname = f"{valid_count}.pt"; valid_count += 1
            else:
                subset = "test";  fname = f"{test_count}.pt";  test_count  += 1
            cursor += 1

            base_obj = {
                "img_raw": torch.from_numpy(arr01).float(),
                "edis_raw": torch.from_numpy(sd).float(),
                "category": cls_name,
            }
            torch.save(base_obj, os.path.join(out_base, subset, fname))

            if BUILD_CP_MIRROR:
                if cls_name == TARGET_CLASS and face is None and len(face_library) > 0:
                    comp = random.choice(face_library)
                    aug_raw, aug_dist = paste_component_into_target(arr01, comp)
                    if aug_dist is None:
                        cp_obj = base_obj
                    else:
                        cp_obj = {
                            "img_raw": torch.from_numpy(aug_raw).float(),
                            "edis_raw": torch.from_numpy(aug_dist).float(),
                            "category": cls_name,
                            "cp_from": "face_library",
                        }
                else:
                    cp_obj = base_obj
                torch.save(cp_obj, os.path.join(out_cp, subset, fname))

    print("Done.")
    print(f"train={train_count}, valid={valid_count}, test={test_count}")
    if BUILD_CP_MIRROR:
        print(f"Original at: {out_base} ; CP-mirror at: {out_cp}")
