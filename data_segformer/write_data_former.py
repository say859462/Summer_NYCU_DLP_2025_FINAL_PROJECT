import os
import ujson as json
import numpy as np
from PIL import Image, ImageDraw
import torch
import random
from scipy.ndimage import distance_transform_edt

# --- 1. 設定 (單一檔案 & 包含距離場) ---
# 指定單一輸入檔案路徑
# 請根據您的環境修改此路徑
input_file_path = r"SPG\\Perceptual Grouping\\airplane.ndjson"

# 輸出資料夾
output_dir = "data_former_pt"
os.makedirs(output_dir, exist_ok=True)
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 圖像與距離場設定
img_size = 156
padded_size = 256

# --- 2. 讀取與切分資料 (還原原版切分邏輯) ---
print(f"正在從 {input_file_path} 讀取資料...")
# 檢查副檔名，使其能同時處理 .json 和 .ndjson
file_ext = os.path.splitext(input_file_path)[1]
if file_ext not in [".json", ".ndjson"]:
    raise ValueError(f"不支援的檔案格式: {file_ext}")

with open(input_file_path, "r", encoding="utf8") as fp:
    json_data = json.load(fp)

# 從檔名中提取類別名稱
category_name = os.path.splitext(os.path.basename(input_file_path))[0]

# 將 (sketch 數據, 類別名稱) 存入 list
sketches_with_category = []
for sketch in json_data["train_data"]:
    sketches_with_category.append((sketch, category_name))

random.shuffle(sketches_with_category)

# 採用原版的固定數量切分法
train_split_count = 700
train_data = sketches_with_category[:train_split_count]
test_data = sketches_with_category[train_split_count:]

print(f"資料讀取完成。")
print(f"訓練集: {len(train_data)} 筆, 測試集: {len(test_data)} 筆。")


# --- 3. 輔助函式 ---
def get_bounds(data, factor=1):
    min_x, max_x, min_y, max_y = 0, 0, 0, 0
    abs_x, abs_y = 0, 0
    for i in range(len(data)):
        x, y = float(data[i][0]) / factor, float(data[i][1]) / factor
        abs_x += x
        abs_y += y
        min_x, min_y = min(min_x, abs_x), min(min_y, abs_y)
        max_x, max_y = max(max_x, abs_x), max(max_y, abs_y)
    return (min_x, max_x, min_y, max_y)


def scale_bound(stroke, average_dimension=img_size):
    bounds = get_bounds(stroke, 1)
    max_dimension = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    if max_dimension == 0:
        return np.array(stroke)
    stroke = np.array(stroke)
    scale = max_dimension / average_dimension
    stroke[:, 0:2] = stroke[:, 0:2] / scale
    return stroke


def strokes_to_lines(strokes):
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
    result = [[] for _ in range(num_groups)]
    for i, num in enumerate(lst):
        if num != -1 and num < num_groups:
            result[num].append(i)
    return result


# --- 4. 資料處理與寫入函式 ---
def process_and_write_data(dataset, subset_path, subset_name):
    print(f"正在處理並寫入 {subset_name} 資料...")
    for idx, (sketch_data, category_name) in enumerate(dataset):
        lines, group_id = strokes_to_lines(sketch_data)

        valid_group_ids = [gid for gid in group_id if gid != -1]
        if not valid_group_ids:
            continue

        nb_stroke = len(lines)
        nb_group = max(valid_group_ids) + 1

        index_group = find_duplicate_indices(group_id, nb_group)
        glabel = np.zeros((nb_group, nb_stroke), dtype=np.int64)
        for row, row_indices in enumerate(index_group):
            for col in row_indices:
                glabel[row][col] = 1

        stroke_images = []
        stroke_distance_fields = []
        for line in lines:
            img = Image.new("1", (img_size, img_size), 0)
            draw = ImageDraw.Draw(img)
            if len(line) >= 2:
                pixels = [(int(x), int(y)) for x, y in line]
                draw.line(pixels, fill=1, width=2)

            # img = img.transpose(Image.FLIP_TOP_BOTTOM)
            arr = np.array(img, dtype=np.float32)

            arr_with_pad = np.zeros((padded_size, padded_size), dtype=np.float32)
            start_pos = (padded_size - img_size) // 2
            arr_with_pad[
                start_pos : start_pos + img_size, start_pos : start_pos + img_size
            ] = arr
            stroke_images.append(arr_with_pad)

            # # --- 計算距離場 ---
            # inverted_arr = 1.0 - arr_with_pad
            # euclidean_distance = distance_transform_edt(inverted_arr)
            # k = 0.001
            # with np.errstate(over="ignore"):
            #     distance_field = 1.0 / (1.0 + k * np.exp(euclidean_distance))
            # stroke_distance_fields.append(distance_field)

        img_raw_np = np.stack(stroke_images, axis=-1)
        edis_raw_np = np.stack(stroke_distance_fields, axis=-1)

        # 儲存的字典包含圖像、距離場、標籤和類別
        data_to_save = {
            "img_raw": torch.from_numpy(img_raw_np).float(),
            "glabel_raw": torch.from_numpy(glabel).long(),
            # "category": category_name,  # 你的改進，可以保留
        }

        torch.save(data_to_save, os.path.join(subset_path, f"{idx}.pt"))

    print(f"{subset_name} 資料寫入完成。")


# --- 5. 執行處理 ---
process_and_write_data(train_data, train_dir, "train")
process_and_write_data(test_data, test_dir, "test")

print("所有流程執行完畢。")
