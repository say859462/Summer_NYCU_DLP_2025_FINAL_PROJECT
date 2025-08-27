import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import logging
import albumentations as A
import random
import numpy as np
from PIL import Image

# --- 關鍵修正：確保匯入 ReplayCompose ---
from albumentations.core.composition import ReplayCompose

loader_logger = logging.getLogger("main.loader")


class StrokeAugmentation:
    """Stroke-level 資料增強類別（使用 ReplayCompose 確保變換一致性）"""

    def __init__(self, apply_prob=0.5, img_size=256):
        self.apply_prob = apply_prob
        self.img_size = img_size

        # --- 關鍵修正：使用 ReplayCompose ---
        self.augmentation_pipeline = ReplayCompose(
            [
                A.Affine(
                    scale=(0.8, 1.2),
                    translate_percent=(-0.1, 0.1),
                    rotate=(-15, 15),
                    shear=(-5, 5),
                    interpolation=Image.NEAREST,
                    p=1.0,
                )
            ],
            is_check_shapes=False,
        )

    # __call__ 方法不再需要，邏輯已移至 Dataset 中以處理 replay
    # def __call__(...):


class SketchAugmentation:
    """Sketch-level 資料增強類別（使用 ReplayCompose 確保剛性變換）"""

    def __init__(self, apply_prob=0.5, img_size=256):
        self.apply_prob = apply_prob
        self.img_size = img_size

        # --- 關鍵修正：同樣使用 ReplayCompose ---
        self.geometric_augmentation = ReplayCompose(
            [
                A.Affine(
                    scale=(0.8, 1.2),
                    translate_percent=(-0.15, 0.15),
                    rotate=(-15, 15),
                    shear=(-10, 10),
                    interpolation=Image.NEAREST,
                    p=1.0,
                )
            ],
            is_check_shapes=False,
        )

    def __call__(self, sketch_data):
        if random.random() > self.apply_prob:
            return sketch_data

        img_raw, glabel_raw, nb_stroke, nb_gp = sketch_data

        # 隨機丟棄 strokes
        if nb_stroke > 3 and random.random() < 0.5:
            img_raw, glabel_raw, nb_stroke = self._random_discard_strokes(
                img_raw, glabel_raw, nb_stroke, nb_gp
            )

        img_np = img_raw.numpy()
        H, W, S = img_np.shape

        # --- 關鍵修正：使用 ReplayCompose 確保所有筆劃應用相同變換 ---
        augmented_strokes = []
        replay_data = None

        for s in range(S):
            stroke_img = img_np[:, :, s]

            if np.sum(stroke_img) > 0:
                if replay_data is None:
                    # 對第一個有效筆劃應用增強，並儲存變換參數
                    data = self.geometric_augmentation(image=stroke_img)
                    augmented_stroke = data["image"]
                    replay_data = data["replay"]
                else:
                    # 對後續筆劃應用完全相同的變換
                    data = self.geometric_augmentation.replay(
                        replay_data, image=stroke_img
                    )
                    augmented_stroke = data["image"]
            else:
                augmented_stroke = stroke_img

            augmented_strokes.append(augmented_stroke)

        augmented_img = np.stack(augmented_strokes, axis=-1)

        return torch.from_numpy(augmented_img), glabel_raw, nb_stroke, nb_gp

    def _random_discard_strokes(self, img_raw, glabel_raw, nb_stroke, nb_gp):
        if nb_stroke <= 1:
            return img_raw, glabel_raw, nb_stroke

        drop_ratio = random.uniform(0.1, 0.3)
        drop_count = max(1, int(nb_stroke * drop_ratio))

        if drop_count >= nb_stroke:
            return img_raw, glabel_raw, nb_stroke

        valid_indices = [
            i
            for i in range(min(nb_stroke, img_raw.shape[2]))
            if torch.any(img_raw[:, :, i] > 0)
        ]
        if len(valid_indices) <= drop_count:
            return img_raw, glabel_raw, nb_stroke

        drop_indices = random.sample(valid_indices, drop_count)
        new_img_raw = img_raw.clone()
        new_glabel_raw = glabel_raw.clone()

        for idx in drop_indices:
            new_img_raw[:, :, idx] = 0
            new_glabel_raw[:, idx] = 0

        new_nb_stroke = int(torch.sum(torch.any(new_img_raw > 0, dim=(0, 1))))
        return new_img_raw, new_glabel_raw, new_nb_stroke


class AeDataset(Dataset):
    """用於 Autoencoder 訓練的 PyTorch Dataset。"""

    def __init__(self, data_dir, raw_size, prefix, augment=False):
        super().__init__()
        self.raw_size = raw_size
        self.augment = augment
        self.stroke_aug = StrokeAugmentation(
            apply_prob=0.5 if augment else 0.0, img_size=raw_size[0]
        )
        subset_path = os.path.join(data_dir, prefix)

        if not os.path.isdir(subset_path):
            raise FileNotFoundError(f"找不到資料夾: {subset_path}。")

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
        )
        input_dis = (
            data["edis_raw"].view(self.raw_size[0], self.raw_size[1]).unsqueeze(0)
        )

        # --- 已修正：使用 ReplayCompose 確保變換一致 ---
        if self.augment and random.random() < self.stroke_aug.apply_prob:
            input_raw_np = input_raw.squeeze(0).numpy()
            input_dis_np = input_dis.squeeze(0).numpy()

            # 第一次應用，獲取增強後的圖像和 replay data
            data_raw = self.stroke_aug.augmentation_pipeline(image=input_raw_np)
            aug_input_raw = data_raw["image"]
            replay_data = data_raw["replay"]

            # 第二次應用，使用 replay data 來應用完全相同的變換
            data_dis = self.stroke_aug.augmentation_pipeline.replay(
                replay_data, image=input_dis_np
            )
            aug_input_dis = data_dis["image"]

            # 轉回 tensor
            input_raw = torch.from_numpy(aug_input_raw).unsqueeze(0)
            input_dis = torch.from_numpy(aug_input_dis).unsqueeze(0)

        return input_raw, input_dis


class GPRegDataset(Dataset):
    """用於 Transformer 分割模型訓練的 PyTorch Dataset。"""

    def __init__(self, data_dir, raw_size, prefix, augment=False):
        super().__init__()
        self.raw_size = raw_size
        self.augment = augment
        self.prefix = prefix
        self.sketch_aug = SketchAugmentation(
            apply_prob=0.5 if augment else 0.0, img_size=raw_size[0]
        )
        subset_path = os.path.join(data_dir, prefix)

        if not os.path.isdir(subset_path):
            raise FileNotFoundError(f"找不到資料夾: {subset_path}。")

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
        img_raw = data["img_raw"]
        glabel_raw = data["glabel_raw"]

        #if self.prefix == "train" and glabel_raw.shape[0] > 1:
        #    stroke_counts_per_group = torch.sum(glabel_raw, dim=1)
        #    sorted_indices = torch.argsort(stroke_counts_per_group, descending=True)
        #    glabel_raw = glabel_raw[sorted_indices]
        #    if "part_names" in data:
        #        part_names = data["part_names"]
        #        data["part_names"] = [part_names[i] for i in sorted_indices]

        nb_stroke = img_raw.shape[2]
        nb_gps = glabel_raw.shape[0]

        if self.augment:
            img_raw, glabel_raw, nb_stroke, nb_gps = self.sketch_aug(
                (img_raw, glabel_raw, nb_stroke, nb_gps)
            )

        return img_raw, glabel_raw, int(nb_stroke), int(nb_gps)


def gpreg_collate_fn(batch):
    """用於 GPRegDataset 的自定義 collate_fn。"""
    imgs, glabels, nb_strokes, nb_gps = zip(*batch)
    max_strokes = max(s.shape[2] for s in imgs)
    max_gps = max(g.shape[0] for g in glabels)
    padded_imgs, padded_glabels = [], []
    stroke_padding_value = -2.0
    label_padding_value = -1

    for img, glabel in zip(imgs, glabels):
        pad_strokes_img = max_strokes - img.shape[2]
        padded_img = F.pad(
            img, (0, pad_strokes_img, 0, 0, 0, 0), "constant", stroke_padding_value
        )
        padded_imgs.append(padded_img)

        pad_gps_label = max_gps - glabel.shape[0]
        pad_strokes_label = max_strokes - glabel.shape[1]
        padded_glabel = F.pad(
            glabel,
            (0, pad_strokes_label, 0, pad_gps_label),
            "constant",
            label_padding_value,
        )
        padded_glabels.append(padded_glabel)

    return (
        torch.stack(padded_imgs),
        torch.stack(padded_glabels),
        torch.tensor(nb_strokes, dtype=torch.int32),
        torch.tensor(nb_gps, dtype=torch.int32),
    )
