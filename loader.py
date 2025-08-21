import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import logging

loader_logger = logging.getLogger("main.loader")

class AeDataset(Dataset):
    """
    用於 Autoencoder 訓練的 PyTorch Dataset。
    從資料夾讀取 .pt 檔案。
    """
    def __init__(self, data_dir, raw_size, prefix):
        super().__init__()
        self.raw_size = raw_size
        subset_path = os.path.join(data_dir, prefix)
        
        if not os.path.isdir(subset_path):
            raise FileNotFoundError(f"找不到資料夾: {subset_path}。請先執行 write_data_embed.py。")
            
        self.file_list = [os.path.join(subset_path, f) for f in os.listdir(subset_path) if f.endswith('.pt')]
        loader_logger.info(f"從 {subset_path} 載入 {len(self.file_list)} 個檔案。")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        data = torch.load(self.file_list[index])
        
        input_raw = data['img_raw'].view(self.raw_size[0], self.raw_size[1]).unsqueeze(0) # [1, H, W]
        input_dis = data['edis_raw'].view(self.raw_size[0], self.raw_size[1]).unsqueeze(0) # [1, H, W]

        return input_raw, input_dis

class GPRegDataset(Dataset):
    """
    用於 Transformer 分割模型訓練的 PyTorch Dataset。
    從資料夾讀取 .pt 檔案。
    """
    def __init__(self, data_dir, raw_size, prefix):
        super().__init__()
        self.raw_size = raw_size
        subset_path = os.path.join(data_dir, prefix)

        if not os.path.isdir(subset_path):
            raise FileNotFoundError(f"找不到資料夾: {subset_path}。請先執行 write_data_former.py。")

        self.file_list = [os.path.join(subset_path, f) for f in os.listdir(subset_path) if f.endswith('.pt')]
        loader_logger.info(f"從 {subset_path} 載入 {len(self.file_list)} 個檔案。")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        data = torch.load(self.file_list[index])
        
        img_raw = data['img_raw'] # [H, W, S]
        glabel_raw = data['glabel_raw'] # [G, S]
        
        nb_stroke = glabel_raw.shape[1]
        nb_gp = glabel_raw.shape[0]

        return img_raw, glabel_raw, nb_stroke, nb_gp

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
        padded_img = F.pad(img, (0, pad_strokes, 0, 0, 0, 0), 'constant', stroke_padding_value)
        padded_imgs.append(padded_img)

        pad_gps = max_gps - glabel.shape[0]
        pad_strokes_label = max_strokes - glabel.shape[1]
        padded_glabel = F.pad(glabel, (0, pad_strokes_label, 0, pad_gps), 'constant', label_padding_value)
        padded_glabels.append(padded_glabel)

    return (
        torch.stack(padded_imgs),
        torch.stack(padded_glabels),
        torch.tensor(nb_strokes, dtype=torch.int32),
        torch.tensor(nb_gps, dtype=torch.int32)
    )
