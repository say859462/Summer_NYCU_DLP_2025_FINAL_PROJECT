# ContextSeg-PyTorch: 論文《ContextSeg》的 PyTorch 實現

這是一個基於 PyTorch 的 [ContextSeg: Sketch Semantic Segmentation by Querying the Context with Attention](https://arxiv.org/abs/2311.16682) 論文實現版本。

此專案將原始的 TensorFlow 程式碼完全轉換為 PyTorch，並進行了以下主要修改：

- **框架轉換**：所有模型架構、訓練迴圈和資料載入器都已從 TensorFlow/Keras 轉換為 PyTorch。
- **依賴性簡化**：將距離場計算從 GeodisTK 函式庫替換為 `scipy`，大幅簡化了環境設定的複雜性。
- **資料格式更新**：移除對 `tfrecord` 的依賴，改用 PyTorch 原生的 `.pt` 格式來儲存預處理後的資料，徹底解決了 TensorFlow 和相關套件的版本衝突問題。
- **額外實作**：相較於 source code，paper 提到很多原本 code 沒有實作的細節，例如 loss function 的定義、DF k 值選擇、以及 data augmentation，在這裡都將其實做出來，後續經由實驗選擇較好者。train_segformer.py增加focal loss，scheduled sampling

## 專案結構

```
.
├── data/              # (手動建立) 存放原始 .ndjson 資料集
├── data_embed_pt/     # (腳本生成) 存放嵌入網路的 .pt 訓練資料
├── data_former_pt/    # (腳本生成) 存放分割網路的 .pt 訓練資料
├── result/            # (腳本生成) 存放訓練結果、模型權重和日誌
├── README.md          # 本文件
├── network.py         # PyTorch 模型架構
├── loader.py          # PyTorch 資料載入器
├── write_data_embed.py  # 預處理腳本 (階段一)
├── write_data_former.py # 預處理腳本 (階段二)
├── train_Embed.py     # 訓練腳本 (階段一)
└── train_Segformer.py   # 訓練腳本 (階段二)
```

## 環境設定

建議使用 Python 3.11 或更新版本。

您可以使用 `pip` 來安裝所有必要的 Python 套件：

```bash
pip install torch torchvision numpy == 1.26.4  Pillow ujson opencv-python==4.8.1.78  albumentations
```

> **註**：建議安裝 NumPy 1.x 版本 (`<2`) 以確保與某些相依套件的最佳相容性。

## 使用教學

請依照以下步驟來執行完整的資料處理與模型訓練流程。

### 步驟 1：下載並準備資料集

從以下連結下載原始資料集，並將其 `.ndjson` (或 `.json`) 檔案放入您手動建立的 `data/` 資料夾中。

- [SPG Dataset (於 SketchX 中)](https://www.google.com/search?q=http://sketchx.dr-cg.com/share)
- [CreativeSketch Dataset](https://www.google.com/search?q=https://github.com/facebookresearch/CreativeSketch)

### 步驟 2：預處理資料

執行以下兩個腳本來處理原始的資料檔案。它們會讀取 `data/` 中的檔案，並在 `data_embed_pt/` 和 `data_former_pt/` 中生成 PyTorch (`.pt`) 格式的訓練檔案。

```bash
# 為第一階段的嵌入網路準備資料 (合併所有類別)
python write_data_embed.py

# 為第二階段的分割網路準備資料 (逐一類別)
python write_data_former.py
```

### 步驟 3：訓練筆劃嵌入網路 (階段一)

執行以下指令來訓練第一階段的筆劃嵌入網路 (Embedding Network)。這個網路的目的是學習如何將筆劃圖像轉換為有意義的特徵向量。

```bash
python train_Embed.py --dbDir data_embed_pt --outDir result --status train
```

- `--dbDir`: 指定包含 `.pt` 檔案的資料夾。
- `--outDir`: 指定儲存模型權重、日誌和結果的資料夾。
- `--status`: 設定為 `train` 模式。

訓練完成後，最佳的模型權重 (checkpoint) 將會儲存在 `result/` 資料夾下的子目錄中。

### 步驟 4：訓練分割 Transformer (階段二)

使用第一階段產生的權重來訓練第二階段的分割 Transformer。

```bash
python train_Segformer.py --dbDir data_former_pt --outDir result --status train --embed_ckpt path/to/your/embedding_model.pth
```

- `--embed_ckpt`: **(重要)** 請將此參數指向您在步驟 3 中儲存的最佳模型權重檔案的路徑（例如 `result/_2025.../checkpoints/model_step_150000.pth`）。

## 引用

如果您在研究中使用了這個專案，請考慮引用原作者的論文：

```bibtex
@article{wang2023contextseg,
  title={ContextSeg: Sketch Semantic Segmentation by Querying the Context with Attention},
  author={Wang, Jiawei and Li, Changjian},
  journal={arXiv preprint arXiv:2311.16682},
  year={2023}
}
```
