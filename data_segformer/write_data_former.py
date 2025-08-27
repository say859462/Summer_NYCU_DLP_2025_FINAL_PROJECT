import os
import ujson as json
import numpy as np
from PIL import Image, ImageDraw
import torch
import random

# --- 1. 設定 ---
input_file_path = r"SPG\\Perceptual Grouping\\calculator.ndjson"
output_dir = "data_former_pt"
os.makedirs(output_dir, exist_ok=True)
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

img_size = 156
padded_size = 256

# --- 2. 讀取與切分資料 (80% training, 20% test) ---
print(f"正在從 {input_file_path} 讀取資料...")

with open(input_file_path, "r", encoding="utf8") as fp:
    json_data = json.load(fp)

data_list = json_data["train_data"]
random.shuffle(data_list)

# 使用80%作為training data
split_index = int(len(data_list) * 0.8)
train_data = data_list[:split_index]
test_data = data_list[split_index:]

print(f"資料讀取完成。")
print(f"訓練集: {len(train_data)} 筆, 測試集: {len(test_data)} 筆。")


# --- 3. 輔助函式 ---
def get_bounds(data, factor=1):
    """Return bounds of data."""
    min_x, max_x, min_y, max_y = 0, 0, 0, 0
    abs_x, abs_y = 0, 0

    for i in range(len(data)):
        x = float(data[i][0]) / factor
        y = float(data[i][1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)


def scale_bound(stroke, average_dimension=img_size):
    """Scale an entire image to be less than a certain size."""
    bounds = get_bounds(stroke, 1)
    max_dimension = max(bounds[1] - bounds[0], bounds[3] - bounds[2])

    if max_dimension == 0:
        return np.array(stroke)

    stroke = np.array(stroke)
    scale = max_dimension / average_dimension
    stroke[:, 0:2] = stroke[:, 0:2] / scale
    return stroke


def strokes_to_lines(strokes):
    """Convert stroke-3 format to polyline format."""
    strokes = scale_bound(strokes)
    x, y = 0, 0
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
            cur_group_ip = int(strokes[i][3])

    return lines, group_id


def find_duplicate_indices(lst, num_groups):
    """Find indices for each group."""
    result = [[] for _ in range(num_groups)]

    for i, num in enumerate(lst):
        if num != -1 and num < num_groups:
            result[int(num)].append(i)

    return result


# --- 4. 資料處理函式 ---
def process_data(dataset, subset_name):
    print(f"正在處理 {subset_name} 資料...")

    input_raw_list = []
    glabel_raw_list = []
    valid_count = 0

    for idx, line_list in enumerate(dataset):
        try:
            lines, group_id = strokes_to_lines(line_list)

            # 過濾無效的 group_id
            valid_group_ids = [gid for gid in group_id if gid != -1]
            if not valid_group_ids:
                continue

            nb_stroke = len(lines)
            nb_group = max(valid_group_ids) + 1  # 根據實際group數量設置

            index_group = find_duplicate_indices(group_id, nb_group)

            # 創建 group label
            glabel = np.zeros((nb_group, nb_stroke), dtype=np.int64)
            for row, row_indices in enumerate(index_group):
                for col in row_indices:
                    if col < nb_stroke:  # 確保索引不越界
                        glabel[row][col] = 1

            # 創建 stroke 圖像
            stroke_images = []
            for line in lines:
                img = Image.new("1", (img_size, img_size), 0)
                draw = ImageDraw.Draw(img)

                if len(line) >= 2:
                    pixels = [(int(x), int(y)) for x, y in line]
                    draw.line(pixels, fill=1, width=2)

                arr = np.array(img, dtype=np.float32)

                # 填充到 256x256
                arr_with_pad = np.zeros((padded_size, padded_size), dtype=np.float32)
                start_pos = (padded_size - img_size) // 2
                arr_with_pad[
                    start_pos : start_pos + img_size, start_pos : start_pos + img_size
                ] = arr

                stroke_images.append(arr_with_pad)

            # 轉換為正確的形狀 (256, 256, num_strokes)
            img_raw = np.stack(stroke_images, axis=-1)

            input_raw_list.append(img_raw)
            glabel_raw_list.append(glabel)
            valid_count += 1

        except Exception as e:
            print(f"處理第 {idx} 個樣本時出錯: {e}")
            continue

    print(f"{subset_name} 處理完成，有效樣本數: {valid_count}")
    return input_raw_list, glabel_raw_list


# --- 5. 處理訓練和測試數據 ---
train_input_raw, train_glabel_raw = process_data(train_data, "train")
test_input_raw, test_glabel_raw = process_data(test_data, "test")


# --- 6. 保存為 PyTorch 格式 ---
def save_as_pt(input_list, glabel_list, save_dir):
    print(f"正在保存數據到 {save_dir}...")

    for idx, (img_raw, glabel_raw) in enumerate(zip(input_list, glabel_list)):
        data_to_save = {
            "img_raw": torch.from_numpy(img_raw).float(),
            "glabel_raw": torch.from_numpy(glabel_raw).long(),
        }

        torch.save(data_to_save, os.path.join(save_dir, f"{idx}.pt"))

    print(f"已保存 {len(input_list)} 個樣本")


# 保存訓練和測試數據
save_as_pt(train_input_raw, train_glabel_raw, train_dir)
save_as_pt(test_input_raw, test_glabel_raw, test_dir)

print("所有流程執行完畢。")
print(
    f"總計 - 訓練集: {len(train_input_raw)} 個有效樣本, 測試集: {len(test_input_raw)} 個有效樣本"
)
