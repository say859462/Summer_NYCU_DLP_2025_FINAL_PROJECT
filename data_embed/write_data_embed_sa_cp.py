
import os
import ujson as json
import numpy as np
from PIL import Image, ImageDraw
import torch
import random
import GeodisTK
from tqdm import tqdm

# ================= Config =================
TARGET_CLASS = "ant"        # 只對此類別做 face copy-paste
TARGET_LABEL = "face"       # 語意標籤名稱（若原始註記有提供）
BUILD_CP_MIRROR = True      # 產出 semantic-aware 鏡像
RNG_SEED = 123
IMG_SIZE = 256
CENTER_SIZE = 156          # 原本 raster 的內部方塊

# 臉偵測 fallback（沒標籤才會用）
AREA_MIN = 15
AREA_MAX = 1200
HEAD_BAND_FRAC = 0.45

random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

# ------------- 幫手函式：raster 與幾何 -------------
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

def scale_bound(stroke, average_dimension=CENTER_SIZE):
    bounds = get_bounds(stroke)
    max_dimension = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    if max_dimension == 0:
        return np.array(stroke)
    stroke = np.array(stroke)
    stroke[:, 0:2] = stroke[:, 0:2] / (max_dimension / average_dimension)
    return stroke

def strokes_to_lines_and_groups(strokes):
    """strokes: list of [dx, dy, pen, group_id(可選)] or dicts; 回傳 (lines, group_ids)
       group_ids 與 lines 一一對應（每條線一個 group id）"""
    # normalize to array with last col = group_id (default -1)
    norm = []
    for s in strokes:
        if isinstance(s, (list, tuple)) and len(s) >= 3:
            # [dx, dy, pen, group?]
            gid = s[3] if len(s) >= 4 else -1
            norm.append([s[0], s[1], s[2], gid])
        elif isinstance(s, dict):
            dx = s.get("dx") or s.get("x") or 0
            dy = s.get("dy") or s.get("y") or 0
            pen = s.get("pen", 0)
            gid = s.get("group") if "group" in s else s.get("group_id", -1)
            norm.append([dx, dy, pen, gid])
        else:
            # unknown entry, skip
            continue

    norm = scale_bound(norm)
    x = y = 0
    lines, line, group_ids = [], [], []
    cur_gid = -1
    for i in range(len(norm)):
        dx, dy, pen, gid = float(norm[i][0]), float(norm[i][1]), int(norm[i][2]), int(norm[i][3])
        x += dx; y += dy
        line.append([x, y])
        if pen == 1:  # stroke end
            lines.append(line)
            group_ids.append(cur_gid)
            line = []
        else:
            cur_gid = gid
    return lines, group_ids

def group_lines_by_id(lines, group_ids):
    groups = {}
    for line, gid in zip(lines, group_ids):
        groups.setdefault(int(gid), []).append(line)
    return groups  # dict: gid -> list of lines

def rasterize_lines(lines, w=CENTER_SIZE, h=CENTER_SIZE, thick=4):
    img = Image.new("1", (w, h), 0)
    draw = ImageDraw.Draw(img)
    for line in lines:
        if len(line) >= 2:
            draw.line([(int(x), int(y)) for x, y in line], fill=1, width=thick)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return np.array(img, dtype=np.uint8)

def pad_center(arr, H=IMG_SIZE, W=IMG_SIZE):
    out = np.zeros((H, W), dtype=np.uint8)
    sy = (H - arr.shape[0]) // 2
    sx = (W - arr.shape[1]) // 2
    out[sy:sy+arr.shape[0], sx:sx+arr.shape[1]] = arr
    return out

def geodesic_dist(arr01):
    # I: float32 cost image in [0,1]
    img = arr01.astype(np.float32, copy=False)
    # S: uint8 seed/foreground mask {0,1}
    mask = (arr01 > 0.5).astype(np.uint8, copy=False)

    with np.errstate(over="ignore", invalid="ignore"):
        geo_dist = GeodisTK.geodesic2d_raster_scan(img, mask, 0, 5)
        sd = 1.0 / (1.0 + 0.001 * np.exp(geo_dist))
        sd[~np.isfinite(sd)] = 0.0
    return sd.astype(np.float32)

# ------------- CC 與貼上 -------------
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

def paste_component_into_target(target_raw01, comp_mask):
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

# ------------- 類別與標籤擷取 -------------
def infer_category_from_path(path_str: str):
    s = path_str.lower().replace("\\", "/")
    parts = s.split("/")
    # 寬鬆：路徑任何一段包含 ant 就算
    if any("ant" in p for p in parts):
        return "ant"
    return "unknown"

def extract_semantic_groups(raw_json):
    """嘗試從 json 項目中找出 group_id -> label_name 的對應。
       回傳 dict (可為空)。
       支援以下可能：
       - raw_json['groups'] = [{'id':..., 'label':...}, ...]
       - raw_json['labels'] = {'<gid>': 'face', ...}
       - raw_json['semantic'] = [{'group': gid, 'name': 'face'}, ...]
    """
    if isinstance(raw_json, dict):
        if "groups" in raw_json and isinstance(raw_json["groups"], list):
            out = {}
            for g in raw_json["groups"]:
                gid = g.get("id")
                lbl = g.get("label") or g.get("name")
                if gid is not None and lbl is not None:
                    out[int(gid)] = str(lbl).lower()
            if out:
                return out
        if "labels" in raw_json and isinstance(raw_json["labels"], dict):
            try:
                return {int(k): str(v).lower() for k, v in raw_json["labels"].items()}
            except Exception:
                pass
        if "semantic" in raw_json and isinstance(raw_json["semantic"], list):
            out = {}
            for e in raw_json["semantic"]:
                gid = e.get("group") or e.get("gid")
                lbl = e.get("name") or e.get("label")
                if gid is not None and lbl is not None:
                    out[int(gid)] = str(lbl).lower()
            if out:
                return out
    return {}

# ------------- 主流程：讀 → 建 face 庫 → 輸出兩份 -------------
if __name__ == "__main__":
    folder_path = os.path.join("..", "data", "SPG", "Perceptual Grouping")  # 你的原始 JSON/NDJSON 路徑
    out_base = "data_embed_pt"
    out_cp   = "data_embed_pt_sa_cp"

    os.makedirs(out_base, exist_ok=True)
    for s in ["train","valid","test"]:
        os.makedirs(os.path.join(out_base, s), exist_ok=True)
    if BUILD_CP_MIRROR:
        os.makedirs(out_cp, exist_ok=True)
        for s in ["train","valid","test"]:
            os.makedirs(os.path.join(out_cp, s), exist_ok=True)

    # 讀檔
    merged = []
    files  = []
    print("Loading stroke files...")
    for root, _, fnames in os.walk(folder_path):
        for filename in fnames:
            if filename.endswith(".ndjson") or filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                if os.path.getsize(filepath) == 0:
                    continue
                with open(filepath, "r", encoding="utf-8") as f:
                    try:
                        jd = json.load(f)
                    except json.JSONDecodeError:
                        continue
                # 支援兩種格式：{'train_data': [...]} 或直接 list
                items = jd.get("train_data", jd if isinstance(jd, list) else [])
                for it in items:
                    merged.append(it)
                    files.append(filepath)

    if not merged:
        raise RuntimeError("No items found under folder_path. Check your path.")

    # 建立 face 元件庫（優先用語意標籤，其次 fallback）
    face_library = []
    n_items = len(merged)
    print("Building face library (semantic first, heuristic fallback)...")

    for it, path_str in tqdm(zip(merged, files), total=n_items):
        cls_name = infer_category_from_path(path_str)
        if cls_name != TARGET_CLASS:
            continue

        # 解析 lines 與 group ids
        lines, gids = strokes_to_lines_and_groups(it)
        gid_to_label = extract_semantic_groups(it)

        # 先找語意=face 的群組；若沒有，跳到 fallback
        face_lines = []
        if gid_to_label:
            for gid, glines in group_lines_by_id(lines, gids).items():
                if gid in gid_to_label and gid_to_label[gid] == TARGET_LABEL:
                    face_lines.extend(glines)

        if face_lines:
            # rasterize face component mask
            face_arr = rasterize_lines(face_lines, CENTER_SIZE, CENTER_SIZE, thick=4)
            face_arr = pad_center(face_arr, IMG_SIZE, IMG_SIZE)
            face_library.append(face_arr.astype(np.uint8))
        else:
            # fallback: heuristic
            # 先 rasterize whole sketch (groups combined)
            all_arr = rasterize_lines([p for g in group_lines_by_id(lines, gids).values() for p in g],
                                      CENTER_SIZE, CENTER_SIZE, thick=4)
            all_arr = pad_center(all_arr, IMG_SIZE, IMG_SIZE).astype(np.float32)
            # detect small component near head band
            def largest_component_mask(mask_np):
                lab, sizes = connected_components(mask_np)
                if not sizes: return np.zeros_like(mask_np, dtype=np.uint8), lab, sizes, 0
                L = int(np.argmax(sizes)+1)
                return (lab==L).astype(np.uint8), lab, sizes, L
            m = (all_arr > 0.5).astype(np.uint8)
            body, lab, sizes, Lmax = largest_component_mask(m)
            if Lmax != 0:
                x0,y0,x1,y1 = bbox_of_label(lab, Lmax)
                bw, bh = x1-x0+1, y1-y0+1
                if bw >= bh:
                    band_w = max(1, int(round(bw*HEAD_BAND_FRAC)))
                    head = np.zeros_like(m); head[y0:y1+1, x0:x0+band_w] = 1
                    alt  = np.zeros_like(m); alt[y0:y1+1, x1-band_w+1:x1+1] = 1
                else:
                    band_h = max(1, int(round(bh*HEAD_BAND_FRAC)))
                    head = np.zeros_like(m); head[y0:y0+band_h, x0:x1+1] = 1
                    alt  = np.zeros_like(m); alt[y1-band_h+1:y1+1, x0:x1+1] = 1
                cand = []
                for L in range(1, int(max(lab.max(), 0))+1):
                    if L == Lmax: continue
                    sz = sizes[L-1]
                    if sz < AREA_MIN or sz > AREA_MAX: continue
                    ys, xs = np.where(lab==L)
                    if xs.size==0: continue
                    cx, cy = (xs.min()+xs.max())/2, (ys.min()+ys.max())/2
                    if head[int(round(cy)), int(round(cx))] or alt[int(round(cy)), int(round(cx))]:
                        cand.append(L)
                if cand:
                    # 取最大的小元件
                    cand.sort(key=lambda L: sizes[L-1], reverse=True)
                    Lsel = cand[0]
                    comp = (lab==Lsel).astype(np.uint8)
                    face_library.append(comp)

    print(f"[face library] collected {len(face_library)} components for class '{TARGET_CLASS}'")

    # --- 第二輪：輸出兩份資料（原始 + SA-CP 鏡像） ---
    total_samples = 0
    per_item_groups = []
    for it in tqdm(merged, desc="Counting"):
        lines, gids = strokes_to_lines_and_groups(it)
        groups = group_lines_by_id(lines, gids)
        # 每個 group 為一個 sample（依你原本的行為）
        samples = []
        for gid, glines in groups.items():
            arr = rasterize_lines(glines, CENTER_SIZE, CENTER_SIZE, thick=4)
            arr = pad_center(arr, IMG_SIZE, IMG_SIZE).astype(np.float32)
            sd = geodesic_dist(arr)
            samples.append((gid, arr, sd))
        per_item_groups.append((it, samples))
        total_samples += len(samples)

    train_end = int(total_samples * 0.8)
    valid_end = train_end + int(total_samples * 0.1)

    out_counts = {"train":0, "valid":0, "test":0}
    cursor = 0

    print("Writing datasets...")
    for (it, samples), path_str in tqdm(zip(per_item_groups, files), total=len(per_item_groups)):
        cls_name = infer_category_from_path(path_str)
        # 語意映射（若存在）
        gid_to_label = extract_semantic_groups(it)

        for gid, arr01, sd in samples:
            subset = "train" if cursor < train_end else ("valid" if cursor < valid_end else "test")
            fname  = f"{out_counts[subset]}.pt"
            out_counts[subset] += 1
            cursor += 1

            base_obj = {
                "img_raw": torch.from_numpy(arr01).float(),
                "edis_raw": torch.from_numpy(sd).float(),
                "category": cls_name,
                "group_id": int(gid),
            }
            if gid_to_label:
                base_obj["sem_label"] = gid_to_label.get(int(gid), "unknown")

            torch.save(base_obj, os.path.join(out_base, subset, fname))

            # SA-CP 鏡像：若為 ant，且該 sample 沒有 face，就貼一個 face 進去
            if BUILD_CP_MIRROR:
                cp_obj = base_obj
                need_face = (cls_name == TARGET_CLASS)
                if need_face:
                    # 判斷這個 group（或整張）是否已有臉
                    has_face = False
                    if gid_to_label and TARGET_LABEL in gid_to_label.values():
                        # 如果任何 group 被標為 face，我們就視為已有臉
                        has_face = True if gid_to_label.get(int(gid), "") == TARGET_LABEL else TARGET_LABEL in gid_to_label.values()
                    else:
                        # fallback: 用啟發式檢測整張 arr01
                        def has_face_heuristic(raw01):
                            m = (raw01 > 0.5).astype(np.uint8)
                            body, lab, sizes, Lmax = largest_component_mask(m)
                            if Lmax == 0: return False
                            x0,y0,x1,y1 = bbox_of_label(lab, Lmax)
                            bw, bh = x1-x0+1, y1-y0+1
                            if bw >= bh:
                                band_w = max(1, int(round(bw*HEAD_BAND_FRAC)))
                                head = np.zeros_like(m); head[y0:y1+1, x0:x0+band_w] = 1
                                alt  = np.zeros_like(m); alt[y0:y1+1, x1-band_w+1:x1+1] = 1
                            else:
                                band_h = max(1, int(round(bh*HEAD_BAND_FRAC)))
                                head = np.zeros_like(m); head[y0:y0+band_h, x0:x1+1] = 1
                                alt  = np.zeros_like(m); alt[y1-band_h+1:y1+1, x0:x1+1] = 1
                            for L in range(1, int(max(lab.max(), 0))+1):
                                if L == Lmax: continue
                                sz = sizes[L-1]
                                if sz < AREA_MIN or sz > AREA_MAX: continue
                                ys, xs = np.where(lab==L)
                                if xs.size==0: continue
                                cx, cy = (xs.min()+xs.max())/2, (ys.min()+ys.max())/2
                                if head[int(round(cy)), int(round(cx))] or alt[int(round(cy)), int(round(cx))]:
                                    return True
                            return False
                        has_face = has_face_heuristic(arr01)

                    if (not has_face) and len(face_library) > 0:
                        comp = random.choice(face_library)
                        aug_raw, aug_dist = paste_component_into_target(arr01, comp)
                        if aug_dist is not None:
                            cp_obj = {
                                "img_raw": torch.from_numpy(aug_raw).float(),
                                "edis_raw": torch.from_numpy(aug_dist).float(),
                                "category": cls_name,
                                "group_id": int(gid),
                                "cp_from": TARGET_LABEL,
                            }
                            if gid_to_label:
                                cp_obj["sem_label"] = gid_to_label.get(int(gid), "unknown")

                torch.save(cp_obj, os.path.join(out_cp, subset, fname))

    print("Done. Counts:", out_counts)
    if BUILD_CP_MIRROR:
        print(f"Original at: {out_base} ; SA-CP mirror at: {out_cp}")
