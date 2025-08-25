import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import logging
import albumentations as A
import random
import numpy as np
from PIL import Image

loader_logger = logging.getLogger("main.loader")


# Source code 沒有實作的augmentaion 但是paper最後有提到
class StrokeAugmentation:
    """Stroke-level 資料增強類別（只使用 rotation, scale, translation）"""

    def __init__(self, apply_prob=0.5, img_size=256):
        self.apply_prob = apply_prob
        self.img_size = img_size

        # 修正後的 Albumentations 增強管道
        self.augmentation_pipeline = A.Compose(
            [
                A.Affine(
                    scale=(0.8, 1.2),  # 縮放：0.8-1.2倍
                    translate_percent=(-0.1, 0.1),  # 平移：±10%
                    rotate=(-15, 15),  # 旋轉：-15°到+15°
                    shear=(-5, 5),  # 輕微剪切保持流暢性
                    interpolation=Image.NEAREST,  # 最近鄰插值
                    p=0.5,  # 應用概率
                )
            ],
            is_check_shapes=False,
        )  # 避免形狀檢查警告

    def __call__(self, stroke_image):
        """
        對單一 stroke 影像進行增強
        stroke_image: [1, H, W] 的 tensor
        """
        if random.random() > self.apply_prob:
            return stroke_image

        # 轉換為 numpy array
        img_np = stroke_image.squeeze(0).numpy()

        # 應用 Albumentations 增強
        augmented = self.augmentation_pipeline(image=img_np)
        augmented_img = augmented["image"]

        # 轉回 tensor
        return torch.from_numpy(augmented_img).unsqueeze(0)


class SketchAugmentation:
    """Sketch-level 資料增強類別（rotation, scale, translation + random discard strokes）"""

    def __init__(self, apply_prob=0.3, img_size=256):
        self.apply_prob = apply_prob
        self.img_size = img_size

        # Sketch-level 幾何增強管道
        self.geometric_augmentation = A.Compose(
            [
                A.Affine(
                    scale=(0.8, 1.2),  # 縮放：0.7-1.3倍
                    translate_percent=(-0.15, 0.15),  # 平移：±15%
                    rotate=(-15, 15),  # 旋轉：-30°到+30°
                    shear=(-10, 10),  # 輕微剪切
                    interpolation=Image.NEAREST,
                    p=1.0,  # 總是應用幾何變換
                )
            ],
            is_check_shapes=False,
        )

    def __call__(self, sketch_data):
        """
        對整個 sketch 進行增強
        sketch_data: 包含 (img_raw, glabel_raw, nb_stroke, nb_gp) 的 tuple
        """
        if random.random() > self.apply_prob:
            return sketch_data

        img_raw, glabel_raw, nb_stroke, nb_gp = sketch_data

        # 隨機丟棄 strokes（10-30% 的 strokes）
        if nb_stroke > 3 and random.random() < 0.5:
            img_raw, glabel_raw, nb_stroke = self._random_discard_strokes(
                img_raw, glabel_raw, nb_stroke, nb_gp
            )

        # 將 tensor 轉換為 numpy
        img_np = img_raw.numpy()  # [H, W, S]
        H, W, S = img_np.shape

        # 對每個 stroke 單獨應用相同的幾何增強
        augmented_strokes = []
        for s in range(S):
            stroke_img = img_np[:, :, s]

            if np.sum(stroke_img) > 0:  # 只增強非空 stroke
                # 應用幾何增強
                augmented = self.geometric_augmentation(image=stroke_img)
                augmented_stroke = augmented["image"]
            else:
                augmented_stroke = stroke_img

            augmented_strokes.append(augmented_stroke)

        augmented_img = np.stack(augmented_strokes, axis=-1)

        return torch.from_numpy(augmented_img), glabel_raw, nb_stroke, nb_gp

    def _random_discard_strokes(self, img_raw, glabel_raw, nb_stroke, nb_gp):
        """
        隨機丟棄 strokes（10-30% 的 strokes）
        """
        # 如果筆劃數 <= 1，直接返回原始數據
        if nb_stroke <= 1:
            return img_raw, glabel_raw, nb_stroke  # 修正：返回3個值

        # 計算要丟棄的 strokes 數量
        drop_ratio = random.uniform(0.1, 0.3)
        drop_count = max(1, int(nb_stroke * drop_ratio))

        # 確保不丟棄所有筆劃
        if drop_count >= nb_stroke:
            return img_raw, glabel_raw, nb_stroke  # 修正：返回3個值

        # 隨機選擇要丟棄的 stroke 索引
        valid_indices = [
            i
            for i in range(min(nb_stroke, img_raw.shape[2]))
            if torch.any(img_raw[:, :, i] > 0)
        ]
        if len(valid_indices) <= drop_count:
            return img_raw, glabel_raw, nb_stroke  # 修正：返回3個值

        drop_indices = random.sample(valid_indices, drop_count)

        # 創建新的 img_raw 和 glabel_raw
        new_img_raw = img_raw.clone()
        new_glabel_raw = glabel_raw.clone()

        for idx in drop_indices:
            # 清空該 stroke 的圖像數據
            new_img_raw[:, :, idx] = 0
            # 移除該 stroke 的所有標籤
            new_glabel_raw[:, idx] = 0

        # 更新 nb_stroke（僅計算非空筆劃）
        new_nb_stroke = int(torch.sum(torch.any(new_img_raw > 0, dim=(0, 1))))

        return new_img_raw, new_glabel_raw, new_nb_stroke  # 正確：返回3個值


class AeDataset(Dataset):
    """
    用於 Autoencoder 訓練的 PyTorch Dataset。
    從資料夾讀取 .pt 檔案。
    """

    def __init__(self, data_dir, raw_size, prefix, augment=False):
        super().__init__()
        self.raw_size = raw_size
        self.augment = augment
        self.stroke_aug = StrokeAugmentation(
            apply_prob=0.7 if augment else 0.0, img_size=raw_size[0]
        )
        subset_path = os.path.join(data_dir, prefix)

        if not os.path.isdir(subset_path):
            raise FileNotFoundError(
                f"找不到資料夾: {subset_path}。請先執行 write_data_embed.py。"
            )

        self.file_list = [
            os.path.join(subset_path, f)
            for f in os.listdir(subset_path)
            if f.endswith(".pt")
        ]
        loader_logger.info(f"從 {subset_path} 載入 {len(self.file_list)} 個檔案。")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        data = torch.load(self.file_list[index])

        input_raw = (
            data["img_raw"].view(self.raw_size[0], self.raw_size[1]).unsqueeze(0)
        )  # [1, H, W]
        input_dis = (
            data["edis_raw"].view(self.raw_size[0], self.raw_size[1]).unsqueeze(0)
        )  # [1, H, W]

        if self.augment:
            input_raw = self.stroke_aug(input_raw.unsqueeze(0)).squeeze(0)
            input_dis = self.stroke_aug(input_dis.unsqueeze(0)).squeeze(0)
        return input_raw, input_dis


class GPRegDataset(Dataset):
    """
    用於 Transformer 分割模型訓練的 PyTorch Dataset。
    從資料夾讀取 .pt 檔案。
    """

    def __init__(self, data_dir, raw_size, prefix, augment=False):
        super().__init__()
        self.raw_size = raw_size
        self.augment = augment
        self.sketch_aug = SketchAugmentation(
            apply_prob=0.5 if augment else 0.0, img_size=raw_size[0]
        )
        subset_path = os.path.join(data_dir, prefix)

        if not os.path.isdir(subset_path):
            raise FileNotFoundError(
                f"找不到資料夾: {subset_path}。請先執行 write_data_former.py。"
            )

        self.file_list = [
            os.path.join(subset_path, f)
            for f in os.listdir(subset_path)
            if f.endswith(".pt")
        ]
        loader_logger.info(f"從 {subset_path} 載入 {len(self.file_list)} 個檔案。")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        data = torch.load(self.file_list[index])

        img_raw = data["img_raw"]  # [H, W, S]
        glabel_raw = data["glabel_raw"]  # [G, S]

        if (
            self.prefix == "train" and glabel_raw.shape[0] > 1
        ):  # 只在訓練時且有多個組時排序
            # 1. 計算每個組 (glabel_raw 的每一行) 包含的筆劃數
            #    glabel_raw 中，值為 1 表示該筆劃屬於該組
            stroke_counts_per_group = torch.sum(glabel_raw, dim=1)

            # 2. 獲取用於降序排序的索引
            sorted_indices = torch.argsort(stroke_counts_per_group, descending=True)

            # 3. 根據排序後的索引，重新排列 glabel_raw
            glabel_raw = glabel_raw[sorted_indices]

            # 4. (重要) 如果 data 中包含 part_names 列表，也必須一同排序以保持對應關係
            if "part_names" in data:
                part_names = data["part_names"]
                data["part_names"] = [part_names[i] for i in sorted_indices]

        nb_stroke = img_raw.shape[2]  # 直接從形狀獲取
        nb_gps = glabel_raw.shape[0]  # 直接從形狀獲取

        if img_raw.shape[0] != self.raw_size[0] or img_raw.shape[1] != self.raw_size[1]:
            loader_logger.error(
                f"檔案 {self.file_list[index]} 的 img_raw 尺寸無效：{img_raw.shape}"
            )
            raise ValueError(f"無效圖像尺寸：{img_raw.shape}")
        if torch.any(torch.isnan(img_raw)) or torch.any(torch.isinf(img_raw)):
            loader_logger.error(f"檔案 {self.file_list[index]} 的 img_raw 包含無效值")
            raise ValueError("圖像數據包含 NaN 或 Inf")
        if self.augment:
            img_raw, glabel_raw, nb_stroke, nb_gps = self.sketch_aug(
                (img_raw, glabel_raw, nb_stroke, nb_gps)
            )

        return img_raw, glabel_raw, int(nb_stroke), int(nb_gps)


def gpreg_collate_fn(batch):
    """
    用於 GPRegDataset 的自定義 collate_fn。
    處理變長序列的填充 (padding)。
    """
    imgs, glabels, nb_strokes, nb_gps = zip(*batch)

    max_strokes = max(s.shape[2] for s in imgs)
    max_gps = max(g.shape[0] for g in glabels)

    padded_imgs, padded_glabels = [], []

    stroke_padding_value = -2.0
    label_padding_value = -1

    for img, glabel in zip(imgs, glabels):
        pad_strokes = max_strokes - img.shape[2]
        padded_img = F.pad(
            img, (0, pad_strokes, 0, 0, 0, 0), "constant", stroke_padding_value
        )
        padded_imgs.append(padded_img)

        pad_gps = max_gps - glabel.shape[0]
        pad_strokes_label = max_strokes - glabel.shape[1]
        padded_glabel = F.pad(
            glabel, (0, pad_strokes_label, 0, pad_gps), "constant", label_padding_value
        )
        padded_glabels.append(padded_glabel)

    return (
        torch.stack(padded_imgs),
        torch.stack(padded_glabels),
        torch.tensor(nb_strokes, dtype=torch.int32),
        torch.tensor(nb_gps, dtype=torch.int32),
    )
