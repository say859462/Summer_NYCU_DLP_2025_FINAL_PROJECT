import os
import ujson as json
import numpy as np
from PIL import Image, ImageDraw
import torch  # 改為引入 torch
import random
from scipy.ndimage import distance_transform_edt

# --- 資料夾路徑設定 (請根據您的環境修改) ---
folder_path = r"SPG\\Perceptual Grouping"
output_dir = "data_embed_pt"  # 建立新的資料夾以存放 .pt 檔案
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "valid"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

merged_data = []
img_size = 156

print("正在從 .ndjson 檔案讀取資料...")
for filename in os.listdir(folder_path):
    filepath = os.path.join(folder_path, filename)
    if filename.endswith(".ndjson") or filename.endswith(".json"):
        with open(filepath, "r") as json_file:
            try:
                if os.path.getsize(filepath) > 0:
                    json_data = json.load(json_file)
                    if "train_data" in json_data:
                        merged_data.extend(json_data["train_data"])
                else:
                    print(f"警告：跳過空檔案 {filename}")
            except json.JSONDecodeError:
                print(f"警告：無法解析 {filename}，可能檔案格式有誤。已跳過。")

random.shuffle(merged_data)
print(f"資料讀取完成，總共有 {len(merged_data)} 筆 sketches。")


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
            cur_group_ip = strokes[i][3]
    return lines, group_id


def group_lines_by_category(lines, categories):
    category_dict = {}
    for line, category in zip(lines, categories):
        category_dict.setdefault(category, []).append(line)
    return list(category_dict.values())


# --- 資料處理與生成 ---
all_data_pairs = []
print("正在處理 sketches 並生成圖像與距離場...")
for inx, line_list in enumerate(merged_data):
    if (inx + 1) % 1000 == 0:
        print(f"已處理 {inx + 1}/{len(merged_data)}...")

    lines, group_id = strokes_to_lines(line_list)
    grouped_lines = group_lines_by_category(lines, group_id)

    for line_for_a_group in grouped_lines:
        if len(line_for_a_group) < 1:
            continue

        img = Image.new("1", (img_size, img_size), 0)
        draw = ImageDraw.Draw(img)
        for line in line_for_a_group:
            if len(line) < 2:
                continue
            pixels = [(int(x), int(y)) for x, y in line]
            draw.line(pixels, fill=1, width=4)

        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        arr = np.array(img)

        arr_with_pad = np.zeros((256, 256))
        start_row, start_col = (256 - img_size) // 2, (256 - img_size) // 2
        arr_with_pad[
            start_row : start_row + img_size, start_col : start_col + img_size
        ] = arr

        inverted_arr = 1 - arr_with_pad
        euclidean_distance = distance_transform_edt(inverted_arr)
        k = 0.001
        distance_field = 1.0 / (1.0 + k * euclidean_distance)

        all_data_pairs.append((arr_with_pad, distance_field))

# --- 資料分割與寫入 ---
random.shuffle(all_data_pairs)

total_len = len(all_data_pairs)
train_end = int(total_len * 0.8)
valid_end = train_end + int(total_len * 0.1)

train_data = all_data_pairs[:train_end]
valid_data = all_data_pairs[train_end:valid_end]
test_data = all_data_pairs[valid_end:]


def write_pt_files(dataset, subset_name):
    print(f"正在寫入 {subset_name} 資料，共 {len(dataset)} 筆...")
    subset_path = os.path.join(output_dir, subset_name)
    for i, (img_raw, edis_raw) in enumerate(dataset):
        data_to_save = {
            "img_raw": torch.from_numpy(img_raw).float(),
            "edis_raw": torch.from_numpy(edis_raw).float(),
        }
        torch.save(data_to_save, os.path.join(subset_path, f"{i}.pt"))
    print(f"{subset_name} 寫入完成。")


write_pt_files(train_data, "train")
write_pt_files(valid_data, "valid")
write_pt_files(test_data, "test")
