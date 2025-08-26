
# sa_copy_paste.py
# Standalone inter-sketch semantic-aware copy-paste for ContextSeg-style datasets.
# Pure Torch/Numpy implementation; no Albumentations dependency.

from __future__ import annotations
import numpy as np
import torch
from typing import Callable, Optional, Tuple

__all__ = ["sa_copy_paste_apply"]

# ------------------------ utilities ------------------------

def _mask_from_raw01(raw01: torch.Tensor, thr: float = 0.5) -> np.ndarray:
    """
    raw01: (1,H,W) torch.float32 in [0,1]
    return: (H,W) np.uint8 {0,1}
    """
    m = (raw01.detach().squeeze(0).cpu().numpy() > thr).astype(np.uint8)
    return m

def _connected_components_bin(mask_np: np.ndarray):
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
    return labels, sizes  # sizes[i-1] corresponds to label i

def _bbox_of_label(labels: np.ndarray, target_label: int) -> Tuple[int,int,int,int]:
    ys, xs = np.where(labels == target_label)
    if xs.size == 0:
        return (0, 0, labels.shape[1]-1, labels.shape[0]-1)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def _paste_component(raw01: torch.Tensor, labels: np.ndarray, chosen_label: int,
                     dst_xy: Tuple[int,int], current_mask_np: np.ndarray) -> Tuple[torch.Tensor, float]:
    """
    Paste chosen component into raw01 (1,H,W) at dst_xy (left-top). Returns (raw01_new, overlap_ratio).
    """
    H, W = raw01.shape[-2:]
    x0, y0, x1, y1 = _bbox_of_label(labels, chosen_label)
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    dx, dy = dst_xy

    # clip to canvas
    tx0, ty0 = max(0, dx), max(0, dy)
    tx1, ty1 = min(W, dx + bw), min(H, dy + bh)
    if tx0 >= tx1 or ty0 >= ty1:
        return raw01, 1.0  # treat as failure

    sx0, sy0 = tx0 - dx + x0, ty0 - dy + y0
    sx1, sy1 = sx0 + (tx1 - tx0), sy0 + (ty1 - ty0)

    comp = (labels[sy0:sy1, sx0:sx1] == chosen_label).astype(np.uint8)  # h x w

    # overlap ratio with current mask
    dst = np.zeros_like(current_mask_np, dtype=np.uint8)
    dst[ty0:ty1, tx0:tx1] = comp
    overlap = (dst & current_mask_np).sum() / max(1, comp.sum())

    # paste via max (keep strokes)
    patch = torch.from_numpy(comp.astype(np.float32)).to(raw01.device)
    raw01[..., ty0:ty1, tx0:tx1] = torch.maximum(raw01[..., ty0:ty1, tx0:tx1], patch.unsqueeze(0))
    return raw01, float(overlap)

def _chamfer_distance_transform(mask_np: np.ndarray) -> np.ndarray:
    """
    Approximate EDT using chamfer (3-4) transform. Returns normalized [0,1] float array.
    """
    H, W = mask_np.shape
    INF = 10**7
    d = np.full((H, W), INF, np.int32)
    d[mask_np.astype(bool)] = 0

    # forward
    for y in range(H):
        for x in range(W):
            v = d[y, x]
            if y > 0:
                v = min(v, d[y-1, x] + 3)
                if x > 0:   v = min(v, d[y-1, x-1] + 4)
                if x < W-1: v = min(v, d[y-1, x+1] + 4)
            if x > 0:
                v = min(v, d[y, x-1] + 3)
            d[y, x] = v

    # backward
    for y in range(H-1, -1, -1):
        for x in range(W-1, -1, -1):
            v = d[y, x]
            if y < H-1:
                v = min(v, d[y+1, x] + 3)
                if x > 0:   v = min(v, d[y+1, x-1] + 4)
                if x < W-1: v = min(v, d[y+1, x+1] + 4)
            if x < W-1:
                v = min(v, d[y, x+1] + 3)
            d[y, x] = v

    d = d.astype(np.float32)
    if d.max() > 0:
        d /= d.max()
    return d

"""
可進行的 semantic constraint 進階改動

更精細的語意規則：你可以在 pick_template_index 與 fetch_template_raw 外再加一個 choose_component
(template_raw, class_id) 回呼，直接指定要 copy 哪個 component（例如只挑「小而圓」者當眼睛），但這會牽
涉到你如何判定語意部件。

位置先驗：若你有 centerline 或 part priors（例如「眼睛大致在頭 bbox 上半部」），可把 valid_map 再用高
斯/多邊形做出子區域遮罩，傳入 constraint_mask 就能完成。
"""

def _largest_component_mask(mask_np: np.ndarray) -> np.ndarray:
    lab, sizes = _connected_components_bin(mask_np)
    if not sizes: return np.zeros_like(mask_np, dtype=np.uint8)
    L = int(np.argmax(sizes) + 1)
    return (lab == L).astype(np.uint8)

def _valid_positions_from_mask(area_mask: np.ndarray, bw: int, bh: int, keep_inside: bool) -> np.ndarray:
    """回傳所有可行左上角座標 (y,x) 的布林圖。"""
    H, W = area_mask.shape
    if not keep_inside:
        # 只要左上角落在 mask 內即可（較寬鬆）
        return area_mask.astype(bool)
    # 需要整個 component 被容納：連續積分快速檢查
    from numpy.lib.stride_tricks import sliding_window_view
    if H < bh or W < bw:
        return np.zeros_like(area_mask, dtype=bool)
    # 以視窗和積分判斷窗口內是否全為1
    win = sliding_window_view(area_mask, (bh, bw))
    full = (win.sum(axis=(-2, -1)) == (bh * bw))
    # pad 回原尺寸對齊左上角
    pad = np.zeros_like(area_mask, dtype=bool)
    pad[:full.shape[0], :full.shape[1]] = full
    return pad

def _sample_dst_inside_mask(valid_map: np.ndarray, rng: np.random.Generator) -> Optional[Tuple[int,int]]:
    ys, xs = np.where(valid_map)
    if ys.size == 0: return None
    k = rng.integers(0, ys.size)
    return int(xs[k]), int(ys[k])  # (dx, dy)


# --------------------- public API ---------------------

def sa_copy_paste_apply(
    idx: int,
    raw01: torch.Tensor,
    dis01: torch.Tensor,
    fetch_template_raw: Callable[[int], torch.Tensor],
    pick_template_index: Callable[[int], int],
    *,
    prob: float = 0.5,
    min_comp_px: int = 20,
    max_overlap: float = 0.30,
    max_trials: int = 10,
    same_class_only: bool = False,  # hint only; actual class check happens in pick_template_index
    # 語意 constraint
    constraint: str = "none",                 # "none" | "inside_largest" | "provided_mask"
    constraint_mask: Optional[np.ndarray] = None,  # (H,W) uint8/bool
    keep_inside: bool = True,                 # component 必須完整落在語意區域中
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform one inter-sketch copy-paste with probability `prob`.
    - idx: target sample index
    - raw01, dis01: (1,H,W) tensors in [0,1] for the target sample
    - fetch_template_raw(tidx) -> (1,H,W) template raw tensor (float01)
    - pick_template_index(idx) -> tidx (same/different class policy handled by caller)

    Returns: (raw_aug, dis_aug). If not applied, returns originals.
    """
    if prob <= 0.0:
        return raw01, dis01
    if np.random.rand() > prob:
        return raw01, dis01

    H, W = raw01.shape[-2:]
    tidx = pick_template_index(idx)
    if tidx == idx:
        return raw01, dis01

    try:
        t_raw01 = fetch_template_raw(tidx)
        if t_raw01 is None:
            return raw01, dis01
        # ensure correct shape/type
        if t_raw01.dim() == 2:
            t_raw01 = t_raw01.unsqueeze(0)
        t_raw01 = t_raw01.to(raw01.dtype).to(raw01.device).clamp(0, 1)
    except Exception:
        return raw01, dis01

    tm = _mask_from_raw01(t_raw01)
    tlab, tsizes = _connected_components_bin(tm)
    if len(tsizes) == 0:
        return raw01, dis01

    # choose a component by size threshold
    order = np.argsort(tsizes)[::-1]
    chosen = None
    for idx2 in order:
        if tsizes[idx2] >= int(min_comp_px):
            chosen = idx2 + 1
            break
    if chosen is None:
        chosen = order[0] + 1

    x0, y0, x1, y1 = _bbox_of_label(tlab, chosen)
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    sm = _mask_from_raw01(raw01)

    # try `max_trials` random placements (uniform over the canvas)
    rng = np.random.default_rng()
    raw_aug = raw01.clone()

    # --- NEW: build semantic constraint map when needed ---
    if constraint == "inside_largest":
        target_mask = _mask_from_raw01(raw01)
        area_mask = _largest_component_mask(target_mask)
    elif constraint == "provided_mask" and constraint_mask is not None:
        area_mask = (constraint_mask.astype(np.uint8) > 0).astype(np.uint8)
    else:
        area_mask = None  # "none"

    # 可行位置圖（左上角）
    valid_map = None
    if area_mask is not None:
        valid_map = _valid_positions_from_mask(area_mask, bw, bh, keep_inside=keep_inside)

    for _ in range(int(max_trials)):
        if valid_map is None:
            dx = rng.integers(0, max(1, W - bw))
            dy = rng.integers(0, max(1, H - bh))
        else:
            sampled = _sample_dst_inside_mask(valid_map, rng)
            if sampled is None:
                break
            dx, dy = sampled

        raw_aug, overlap = _paste_component(raw_aug, tlab, chosen, (dx, dy), sm)
        if overlap <= float(max_overlap):
            new_mask = (raw_aug.detach().squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)
            new_dist = _chamfer_distance_transform(new_mask)
            dis_aug = torch.from_numpy(new_dist).to(raw_aug.device).unsqueeze(0).to(raw01.dtype)
            return raw_aug, dis_aug

    # failed to place
    return raw01, dis01
