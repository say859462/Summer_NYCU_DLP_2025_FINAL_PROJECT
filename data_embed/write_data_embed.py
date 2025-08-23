import os
import ujson as json
import numpy as np
from PIL import Image, ImageDraw
import torch
import random
from multiprocessing import Pool, cpu_count
import GeodisTK
from tqdm import tqdm
import gc


# --- 工具函式（保持完全一致）---
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


def process_one(line_list):
    """單筆處理，生成影像與 geodesic distance"""
    result_pairs = []
    lines, group_id = strokes_to_lines(line_list)
    grouped_lines = group_lines_by_category(lines, group_id)
    for line_for_a_group in grouped_lines:
        if len(line_for_a_group) == 1:
            continue

        # 生成影像
        img = Image.new("1", (156, 156), 0)
        draw = ImageDraw.Draw(img)
        for line in line_for_a_group:
            if len(line) < 2:
                continue
            draw.line([(int(x), int(y)) for x, y in line], fill=1, width=4)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        arr = np.array(img, dtype=np.uint8)

        # padding
        arr_pad = np.zeros((256, 256), dtype=np.uint8)
        start_row = (256 - 156) // 2
        start_col = (256 - 156) // 2
        arr_pad[start_row : start_row + 156, start_col : start_col + 156] = arr

        # Geodesic distance
        mask = arr_pad.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            geo_dist = GeodisTK.geodesic2d_raster_scan(
                arr_pad.astype(np.float32), mask, 0, 5
            )
            # k: 0.01 source code , k: 0.001 paper indicated
            # sd = 1 / (1 + 0.01 * np.exp(geo_dist))
            sd = 1 / (1 + 0.001 * np.exp(geo_dist))

            sd[np.isinf(sd)] = 0.0
            sd[np.isnan(sd)] = 0.0

        result_pairs.append((arr_pad.astype(np.float32), sd.astype(np.float32)))
    return result_pairs


# ---------------------------- 主程式（邊生成邊儲存版本）----------------------------
if __name__ == "__main__":
    # --- 資料夾設定 ---
    folder_path = r"SPG\\Perceptual Grouping"
    output_dir = "data_embed_pt"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "valid"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    # --- 讀取資料 ---
    merged_data = []
    print("正在從 .ndjson/.json 檔案讀取資料...")
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.endswith(".ndjson") or filename.endswith(".json"):
            if os.path.getsize(filepath) == 0:
                print(f"警告：跳過空檔案 {filename}")
                continue
            with open(filepath, "r") as f:
                try:
                    json_data = json.load(f)
                    if "train_data" in json_data:
                        merged_data.extend(json_data["train_data"])
                except json.JSONDecodeError:
                    print(f"警告：無法解析 {filename}，已跳過")

    random.shuffle(merged_data)
    print(f"資料讀取完成，總共有 {len(merged_data)} 筆 sketches。")

    # --- 預先計算總樣本數以便正確分割 ---
    print("正在預計算總樣本數量...")
    total_samples = 0
    for sketch in tqdm(merged_data):
        lines, group_id = strokes_to_lines(sketch)
        grouped_lines = group_lines_by_category(lines, group_id)
        for line_for_a_group in grouped_lines:
            if len(line_for_a_group) > 1:  # 只計算有效的群組
                total_samples += 1

    print(f"預計生成 {total_samples} 個訓練樣本")

    # 設定分割點
    train_end = int(total_samples * 0.8)
    valid_end = train_end + int(total_samples * 0.1)

    # 計數器
    train_count, valid_count, test_count = 0, 0, 0

    # --- 分批處理並立即儲存 ---
    batch_size = 100  # 更小的批次大小減少記憶體壓力
    num_batches = (len(merged_data) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(merged_data))
        batch_data = merged_data[start_idx:end_idx]

        print(f"處理批次 {batch_idx + 1}/{num_batches} ({len(batch_data)} 個sketches)")

        # 處理當前批次
        with Pool(processes=max(1, cpu_count() - 1)) as pool:
            batch_results = list(pool.imap(process_one, batch_data))

        # 立即儲存當前批次的結果
        for result in batch_results:
            for img_raw, dis_raw in result:
                # 決定屬於哪個資料集
                if train_count < train_end:
                    subset = "train"
                    filename = os.path.join(output_dir, "train", f"{train_count}.pt")
                    train_count += 1
                elif valid_count < (valid_end - train_end):
                    subset = "valid"
                    filename = os.path.join(output_dir, "valid", f"{valid_count}.pt")
                    valid_count += 1
                else:
                    subset = "test"
                    filename = os.path.join(output_dir, "test", f"{test_count}.pt")
                    test_count += 1

                # 立即儲存單一樣本
                torch.save(
                    {
                        "img_raw": torch.from_numpy(img_raw).float(),
                        "edis_raw": torch.from_numpy(dis_raw).float(),
                    },
                    filename,
                )

        # 釋放記憶體
        del batch_results, batch_data
        gc.collect()

        print(
            f"進度: 訓練集 {train_count}/{train_end}, 驗證集 {valid_count}/{valid_end-train_end}, 測試集 {test_count}/{total_samples-valid_end}"
        )

    print("所有資料處理完成！")
    print(f"最終統計: 訓練集 {train_count}, 驗證集 {valid_count}, 測試集 {test_count}")
