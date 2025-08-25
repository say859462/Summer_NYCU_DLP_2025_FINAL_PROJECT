from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
from PIL import Image

# 從您提供的 PyTorch 檔案中匯入模듈
from loader import GPRegDataset, gpreg_collate_fn
from network import GpTransformer, AutoencoderEmbed

# ====== 參數解析  ======
parser = argparse.ArgumentParser(
    description="訓練 Sketch Semantic Segmentation Transformer 模型"
)
parser.add_argument(
    "--status",
    help="執行模式 (train 或 test)",
    type=str,
    default="train",
    choices=["train", "test"],
)
parser.add_argument(
    "--dbDir",
    help="由 write_data*.py 產生的 .pt 資料庫目錄",
    type=str,
    default="data_former_pt",
)
parser.add_argument(
    "--outDir", help="輸出結果的根目錄", type=str, default="result_segformer"
)
parser.add_argument(
    "--embed_ckpt",
    help="預訓練 Embedding Autoencoder 的 .pth checkpoint 路徑",
    type=str,
    default="result\\best.pth",
)
parser.add_argument(
    "--ckpt",
    help="Segformer 模型的 .pth checkpoint 路徑 (若要繼續訓練或測試)",
    type=str,
    default=None,
)
parser.add_argument("--cnt", help="是否從 ckpt 繼續訓練", action="store_true")
parser.add_argument("--maxIter", help="最大訓練步數", type=int, default=1500000)
parser.add_argument("--batchSize", help="批次大小 (實際)", type=int, default=8)
parser.add_argument(
    "--accum_steps",
    help="梯度累積步數，等效 batch size = batchSize * accum_steps",
    type=int,
    default=1,
)
parser.add_argument("--lr", help="學習率", type=float, default=1e-4)
parser.add_argument("--d_model", help="模型的特徵維度", type=int, default=256)
parser.add_argument("--num_layers", help="Transformer 層數", type=int, default=4)
parser.add_argument("--d_ff", help="前饋網路的中間層維度", type=int, default=2048)
parser.add_argument("--num_heads", help="多頭注意力機制的頭數", type=int, default=4)
parser.add_argument("--drop_rate", help="Dropout 比率", type=float, default=0.4)
parser.add_argument(
    "--num_of_group", help="資料集中的最大群組數量", type=int, default=4
)
parser.add_argument(
    "--teacher_forcing_start", help="Teacher forcing 起始比例", type=float, default=1.0
)
parser.add_argument(
    "--teacher_forcing_end", help="Teacher forcing 結束比例", type=float, default=0.2
)
parser.add_argument(
    "--teacher_forcing_decay_steps",
    help="Teacher forcing 衰減步數",
    type=int,
    default=50000,
)
parser.add_argument(
    "--dispLossStep", help="每隔多少步顯示一次日誌", type=int, default=200
)
parser.add_argument(
    "--exeValStep", help="每隔多少步驗證一次模型", type=int, default=1000
)
parser.add_argument(
    "--saveModelStep", help="每隔多少步儲存一次模型", type=int, default=5000
)


# ====== 輔助函式 ======
def create_masks(inp, tar):
    device = inp.device
    enc_padding_mask = (
        (torch.sum(inp, dim=2) == -2.0 * inp.shape[2]).unsqueeze(1).unsqueeze(2)
    )
    dec_padding_mask = enc_padding_mask
    tar_len = tar.shape[1]
    look_ahead_mask = torch.triu(
        torch.ones((tar_len, tar_len), device=device), diagonal=1
    ).bool()
    tar_padding_mask = (
        (torch.sum(tar, dim=2) == -2.0 * tar.shape[2]).unsqueeze(1).unsqueeze(2)
    )
    combined_mask = tar_padding_mask | look_ahead_mask
    return enc_padding_mask, combined_mask, dec_padding_mask


def get_teacher_forcing_ratio(step, start_ratio, end_ratio, decay_steps):
    """
    根據訓練步數計算當前的 teacher forcing ratio
    論文提到從 100% 逐漸衰減到 20%
    """
    if step >= decay_steps:
        return end_ratio

    # 線性衰減
    decay_factor = step / decay_steps
    current_ratio = start_ratio - (start_ratio - end_ratio) * decay_factor
    return current_ratio


def loss_fn(real, pred, gamma=2.0):
    """
    計算 Focal Loss。
    根據論文 Eq. 6，gamma 值預設為 2.0。
    """
    # 步驟 1: 建立遮罩 (mask)，忽略填充值 (-1.0)，這部分與原始程式碼相同
    mask = (real != -1.0).float()

    # 步驟 2: 計算標準的 BCE Loss (但不進行 reduction)
    # F.binary_cross_entropy_with_logits 包含了 sigmoid 操作，更穩定
    bce_loss = F.binary_cross_entropy_with_logits(pred, real.float(), reduction="none")

    # 步驟 3: 計算 Focal Loss 的調變因子 (modulating factor)
    # 首先取得預測機率 p
    p = torch.sigmoid(pred)
    # 根據真實標籤計算 p_t
    p_t = p * real + (1 - p) * (1 - real)
    # 調變因子為 (1 - p_t)^gamma
    modulating_factor = (1.0 - p_t).pow(gamma)

    # 步驟 4: 計算最終的 Focal Loss
    # 將 BCE Loss 與調變因子相乘
    focal_loss = modulating_factor * bce_loss

    # 步驟 5: 應用遮罩並計算平均損失
    masked_loss = focal_loss * mask
    nb_elem = torch.sum(mask)
    if nb_elem == 0:  # 避免除以零
        loss_val = torch.tensor(0.0).to(pred.device)
    else:
        loss_val = torch.sum(masked_loss) / nb_elem

    # --- 準確率計算部分保持不變 ---
    with torch.no_grad():  # 準確率計算不應影響梯度
        pred_sigmoid = torch.sigmoid(pred)
        pred_sigmoid_masked = torch.round(pred_sigmoid) * mask
        real_masked = real * mask

        if nb_elem == 0:
            acc_val = torch.tensor(1.0).to(pred.device)
        else:
            acc_val = 1.0 - (
                torch.sum(torch.abs(pred_sigmoid_masked - real_masked)) / nb_elem
            )

    return loss_val, acc_val


# =============================================================================


def sacc(pred_sigmoid, gt_label):
    mask = (gt_label != -1.0).any(dim=1, keepdim=True)
    gt_label_idx = torch.argmax(gt_label, dim=1)
    pred_label_idx = torch.argmax(torch.round(pred_sigmoid), dim=1)
    correct = (gt_label_idx == pred_label_idx).float()
    correct_masked = correct * mask.squeeze(1).float()
    sacc_val = (
        torch.sum(correct_masked) / torch.sum(mask)
        if torch.sum(mask) > 0
        else torch.tensor(1.0)
    )
    return sacc_val


def cacc(pred_sigmoid, gt_label):
    pred = torch.round(pred_sigmoid[0]).detach().cpu().numpy()
    gt = gt_label[0].detach().cpu().numpy()
    valid_strokes_mask = gt[0, :] != -1.0
    if not np.any(valid_strokes_mask):
        return 0.0
    gt, pred = gt[:, valid_strokes_mask], pred[:, valid_strokes_mask]
    valid_groups_mask = np.sum(gt, axis=1) != -1.0 * gt.shape[1]
    if not np.any(valid_groups_mask):
        return 0.0
    gt, pred = gt[valid_groups_mask, :], pred[valid_groups_mask, :]
    if gt.shape[1] == 0:
        return 0.0
    category_list, prediction_list = np.argmax(gt, axis=0), np.argmax(pred, axis=0)
    group_ids, group_counts = np.unique(category_list, return_counts=True)
    correct_groups = 0
    for gid, total in zip(group_ids, group_counts):
        correct_mask = (category_list == gid) & (prediction_list == gid)
        if (np.sum(correct_mask) / total) >= 0.75:
            correct_groups += 1
    return correct_groups / len(group_ids) if len(group_ids) > 0 else 0.0


def group_images(images, labels):
    images = images.permute(2, 0, 1)
    grouped_images = []
    num_groups = labels.shape[0]
    for i in range(num_groups):
        indices = torch.where(labels[i] == 1)[0]
        if len(indices) > 0:
            grouped_images.append(
                torch.sum(torch.index_select(images, 0, indices), dim=0)
            )
        else:
            grouped_images.append(torch.zeros_like(images[0]))
    return torch.clamp(torch.stack(grouped_images), 0, 1)


def cook_raw(batch_data):
    input_raw, glabel_raw, nb_strokes, nb_gps = batch_data
    input_raw, glabel_raw = input_raw.to(device), glabel_raw.to(device)
    batch_size = input_raw.shape[0]
    input_embeddings, target_embeddings = [], []
    for i in range(batch_size):
        single_input_strokes = (
            input_raw[i, :, :, : nb_strokes[i]].permute(2, 0, 1).unsqueeze(1)
        )
        single_labels = glabel_raw[i, : nb_gps[i], : nb_strokes[i]]
        stroke_embeds = en_modelAESSG.encode(single_input_strokes)
        input_embeddings.append(stroke_embeds)
        grouped_imgs = group_images(input_raw[i, :, :, : nb_strokes[i]], single_labels)
        group_embeds = de_modelAESSG.encode(grouped_imgs.unsqueeze(1))
        start_token = torch.full((1, group_embeds.shape[1]), -1.0, device=device)
        target_embeds_with_start = torch.cat([start_token, group_embeds], dim=0)
        target_embeddings.append(target_embeds_with_start)
    padded_input_embeds = nn.utils.rnn.pad_sequence(
        input_embeddings, batch_first=True, padding_value=-2.0
    )
    padded_target_embeds = nn.utils.rnn.pad_sequence(
        target_embeddings, batch_first=True, padding_value=-2.0
    )
    return padded_input_embeds, padded_target_embeds, glabel_raw


def generate_grouped_image(stroke_images, group_labels):
    num_groups = group_labels.shape[0]
    H, W = stroke_images.shape[0], stroke_images.shape[1]
    canvas = torch.ones(H, W, 3)
    colors = (
        torch.tensor(
            [
                [0, 255, 255],
                [255, 0, 255],
                [255, 255, 0],
                [0, 255, 0],
                [255, 0, 0],
                [0, 0, 255],
                [255, 128, 0],
                [128, 0, 255],
            ],
            dtype=torch.float32,
        )
        / 255.0
    )
    stroke_images, group_labels = stroke_images.cpu(), group_labels.cpu()

    for i in range(num_groups):
        color = colors[i % len(colors)]
        # 獲取當前組的標籤
        group_mask = group_labels[i : i + 1, :]
        # 創建該組的圖像
        group_img = torch.zeros(H, W)

        # 找到屬於這個組的所有筆劃
        stroke_indices = torch.where(group_mask[0] == 1)[0]
        for idx in stroke_indices:
            if idx < stroke_images.shape[2]:  # 確保索引有效
                group_img = torch.maximum(group_img, stroke_images[:, :, idx])

        mask = group_img.unsqueeze(-1)  # 添加通道維度
        colored_layer = color.view(1, 1, 3) * mask
        canvas = canvas * (1.0 - mask) + colored_layer

    # 確保圖像值在正確範圍內並轉換為uint8
    canvas = torch.clamp(canvas, 0, 1)
    return (canvas * 255).numpy().astype(np.uint8)


def save_image_safe(img_array, filepath):
    """安全保存圖像，確保尺寸和格式正確"""
    # 確保圖像是有效的2D或3D數組
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # 已經是RGB格式
        pass
    elif len(img_array.shape) == 2:
        # 灰度圖轉換為RGB
        img_array = np.stack([img_array] * 3, axis=-1)
    else:
        # 其他情況，取第一個通道或轉換為灰度
        img_array = img_array[:, :, 0] if img_array.shape[2] > 0 else img_array
        img_array = np.stack([img_array] * 3, axis=-1)

    # 確保圖像尺寸合理
    if img_array.shape[0] > 10000 or img_array.shape[1] > 10000:
        # 如果尺寸過大，進行縮放
        scale = 1000 / max(img_array.shape[0], img_array.shape[1])
        new_size = (int(img_array.shape[1] * scale), int(img_array.shape[0] * scale))
        img_array = cv2.resize(img_array, new_size)

    # 保存為JPEG格式避免PNG問題
    if filepath.endswith(".png"):
        filepath = filepath.replace(".png", ".jpg")

    success = cv2.imwrite(filepath, img_array)
    if not success:
        print(f"警告: 無法保存圖像到 {filepath}")
        # 嘗試使用PIL作為備用
        try:
            Image.fromarray(img_array).save(filepath)
            print(f"使用PIL成功保存圖像到 {filepath}")
        except Exception as e:
            print(f"使用PIL保存圖像也失敗: {e}")


# ====== 訓練與測試流程 ======
def test_autoregre_step(batch_data):
    transformer.eval()
    with torch.no_grad():
        input_raw, glabel_raw, nb_strokes, nb_gps = batch_data
        input_raw, glabel_raw = input_raw.to(device), glabel_raw.to(device)
        single_input_strokes = (
            input_raw[0, :, :, : nb_strokes[0]].permute(2, 0, 1).unsqueeze(1)
        )
        inp_embed = en_modelAESSG.encode(single_input_strokes).unsqueeze(0)
        gp_token = torch.full((1, 1, hyper_params["d_model"]), -1.0, device=device)
        all_predictions = []
        nb_max_try = nb_gps[0] if nb_gps[0] > 0 else hyper_params["num_of_group"]
        for _ in range(int(nb_max_try)):
            enc_mask, combined_mask, dec_mask = create_masks(inp_embed, gp_token)
            predictions, _ = transformer(
                inp_embed, gp_token, enc_mask, combined_mask, dec_mask
            )
            last_pred = predictions[:, -1:, :]
            all_predictions.append(last_pred)
            pred_labels_for_group = torch.round(torch.sigmoid(last_pred.squeeze(1)))
            grouped_img = group_images(
                input_raw[0, :, :, : nb_strokes[0]], pred_labels_for_group
            )
            new_token_embed = de_modelAESSG.encode(grouped_img.unsqueeze(1))
            gp_token = torch.cat([gp_token, new_token_embed.unsqueeze(1)], dim=1)
        final_predictions = torch.cat(all_predictions, dim=1)
        loss, acc = loss_fn(glabel_raw, final_predictions)
        sacc_val = sacc(torch.sigmoid(final_predictions), glabel_raw)
        cacc_val = cacc(torch.sigmoid(final_predictions), glabel_raw)
    return loss, acc, sacc_val, cacc_val, final_predictions


def train_net():
    logger.info("--- 開始訓練 ---")
    writer = SummaryWriter(log_dir=os.path.join(output_folder, "summary"))

    # ====== NEW: 建立用於儲存驗證過程圖片的資料夾 ======
    val_img_folder = os.path.join(output_folder, "validation_imgs")
    os.makedirs(val_img_folder, exist_ok=True)

    step = hyper_params.get("start_step", 0)
    accumulation_steps = hyper_params.get("accum_steps", 1)

    # Teacher forcing 參數
    tf_start = hyper_params.get("teacher_forcing_start", 1.0)
    tf_end = hyper_params.get("teacher_forcing_end", 0.2)
    tf_decay_steps = hyper_params.get("teacher_forcing_decay_steps", 100000)

    # 從 checkpoint 恢復最佳分數，如果有的話
    best_score_sum = hyper_params.get("best_score_sum", 0.0)
    if best_score_sum > 0:
        logger.info(f"從 checkpoint 恢復最佳分數: {best_score_sum:.6f}")

    if accumulation_steps > 1:
        logger.info(f"啟用梯度累積，步數為: {accumulation_steps}")
        effective_batch_size = hyper_params["batchSize"] * accumulation_steps
        logger.info(
            f"實際批次大小: {hyper_params['batchSize']}, 等效批次大小: {effective_batch_size}"
        )

    train_iter = iter(train_loader)
    optimizer.zero_grad()

    while step < hyper_params["maxIter"]:
        try:
            batch_data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch_data = next(train_iter)

        transformer.train()
        inp_embed, tar_embed, labels = cook_raw(batch_data)

        # 獲取目標序列長度（不包括start token）
        target_seq_len = tar_embed.shape[1] - 1  # 減去start token

        # 獲取當前 teacher forcing ratio
        tf_ratio = get_teacher_forcing_ratio(step, tf_start, tf_end, tf_decay_steps)

        # 決定是否使用 teacher forcing
        use_teacher_forcing = True
        if tf_ratio < 1.0:  # 只有在需要混合時才進行
            use_teacher_forcing = torch.rand(1).item() < tf_ratio

        if use_teacher_forcing:
            # 使用 ground truth 作為 decoder 輸入
            tar_for_input = tar_embed[:, :-1, :]  # 移除最後一個token作為輸入
        else:
            # 使用模型預測作為 decoder 輸入 (auto-regressive)
            batch_size = inp_embed.shape[0]

            # 初始化 decoder 輸入 (只有 start token)
            decoder_input = tar_embed[:, :1, :]  # start token

            # 逐步生成，但限制最大長度為target_seq_len
            max_gen_steps = min(target_seq_len, hyper_params["num_of_group"])

            for group_idx in range(max_gen_steps):
                enc_mask, combined_mask, dec_mask = create_masks(
                    inp_embed, decoder_input
                )
                predictions, _ = transformer(
                    inp_embed, decoder_input, enc_mask, combined_mask, dec_mask
                )

                # 獲取最後一個預測
                last_pred = predictions[:, -1:, :]
                pred_labels = torch.round(torch.sigmoid(last_pred))

                # 根據預測生成對應的 group embedding
                group_embeddings = []
                for i in range(batch_size):
                    # 獲取當前樣本的原始筆劃數
                    num_original_strokes = batch_data[2][i]

                    # 從原始輸入中提取 strokes
                    stroke_images = batch_data[0][i, :, :, :num_original_strokes].to(
                        device
                    )

                    # 將預測標籤裁剪到原始筆劃數，避免越界
                    valid_pred_labels = pred_labels[i][:, :num_original_strokes]

                    # 根據裁剪後的預測標籤生成 group 影像
                    grouped_img = group_images(stroke_images, valid_pred_labels)
                    # 編碼 group 影像
                    group_embed = de_modelAESSG.encode(grouped_img.unsqueeze(1))
                    group_embeddings.append(group_embed)

                # 拼接 group embeddings
                new_tokens = torch.stack(group_embeddings, dim=0)
                decoder_input = torch.cat([decoder_input, new_tokens], dim=1)

            # 確保decoder_input的長度正確（移除start token用於輸入）
            if decoder_input.shape[1] > 1:
                tar_for_input = decoder_input[:, 1:, :]  # 移除start token
            else:
                # 如果只有start token，創建一個dummy輸入
                tar_for_input = torch.full(
                    (batch_size, 1, hyper_params["d_model"]), -1.0, device=device
                )

        # 確保tar_for_input和labels的序列長度一致
        min_seq_len = min(tar_for_input.shape[1], labels.shape[1])
        tar_for_input = tar_for_input[:, :min_seq_len, :]
        labels_trimmed = labels[:, :min_seq_len, :]

        # 生成masks和進行前向傳播
        enc_mask, combined_mask, dec_mask = create_masks(inp_embed, tar_for_input)

        predictions, _ = transformer(
            inp_embed, tar_for_input, enc_mask, combined_mask, dec_mask
        )

        # 確保predictions和labels的形狀完全一致
        if predictions.shape[1] != labels_trimmed.shape[1]:
            min_len = min(predictions.shape[1], labels_trimmed.shape[1])
            predictions = predictions[:, :min_len, :]
            labels_trimmed = labels_trimmed[:, :min_len, :]

        # 額外檢查最後一維
        if predictions.shape[2] != labels_trimmed.shape[2]:
            min_last_dim = min(predictions.shape[2], labels_trimmed.shape[2])
            predictions = predictions[:, :, :min_last_dim]
            labels_trimmed = labels_trimmed[:, :, :min_last_dim]

        # 使用 Focal Loss (gamma=2.0)
        loss, acc = loss_fn(labels_trimmed, predictions)

        loss = loss / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if step % hyper_params["dispLossStep"] == 0:
            sacc_val = sacc(torch.sigmoid(predictions), labels_trimmed)
            cacc_val = cacc(torch.sigmoid(predictions), labels_trimmed)
            # 獲取當前學習率
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"步驟 [{step}/{hyper_params['maxIter']}], "
                f"損失: {loss.item() * accumulation_steps:.6f}, "
                f"Acc: {acc.item():.6f}, "
                f"SAcc: {sacc_val.item():.6f}, "
                f"CAcc: {cacc_val:.6f}, "
                f"TF Ratio: {tf_ratio:.3f}, "
                f"LR: {current_lr:.1e}"
            )
            writer.add_scalar("Loss/train", loss.item() * accumulation_steps, step)
            writer.add_scalar("SAcc/train", sacc_val.item(), step)
            writer.add_scalar("CAcc/train", cacc_val, step)
            writer.add_scalar("Teacher_Forcing_Ratio", tf_ratio, step)
            writer.add_scalar("Learning_Rate", current_lr, step)

        if step > 0 and step % hyper_params["exeValStep"] == 0:
            val_loss, val_sacc, val_cacc, count = 0, 0, 0, 0
            test_iter = iter(test_loader)
            while True:
                try:
                    batch_data_val = next(test_iter)
                    loss_v, _, sacc_v, cacc_v, final_predictions_val = (
                        test_autoregre_step(batch_data_val)
                    )
                    val_loss += loss_v.item()
                    val_sacc += sacc_v.item()
                    val_cacc += cacc_v
                    if count < 3:
                        input_raw, glabel_raw, nb_strokes, _ = batch_data_val
                        n = (
                            int(nb_strokes[0].item())
                            if torch.is_tensor(nb_strokes[0])
                            else int(nb_strokes[0])
                        )

                        gt_vis_img = generate_grouped_image(
                            input_raw[0, :, :, :n], glabel_raw[0]
                        )
                        pred_vis_img = generate_grouped_image(
                            input_raw[0, :, :, :n],
                            torch.round(torch.sigmoid(final_predictions_val[0])),
                        )

                        def to_uint8(img):
                            if torch.is_tensor(img):
                                img = img.detach().cpu().numpy()
                            if img.max() <= 1.0:
                                img = (img * 255).astype(np.uint8)
                            else:
                                img = img.astype(np.uint8)
                            return img

                        gt_vis_img = to_uint8(gt_vis_img)
                        pred_vis_img = to_uint8(pred_vis_img)

                        os.makedirs(val_img_folder, exist_ok=True)
                        save_image_safe(
                            gt_vis_img,
                            os.path.join(
                                val_img_folder, f"val_step_{step}_sample_{count}_gt.jpg"
                            ),
                        )
                        save_image_safe(
                            pred_vis_img,
                            os.path.join(
                                val_img_folder,
                                f"val_step_{step}_sample_{count}_pred.jpg",
                            ),
                        )

                    count += 1
                except StopIteration:
                    break
            avg_loss, avg_sacc, avg_cacc = (
                val_loss / count,
                val_sacc / count,
                val_cacc / count,
            )
            logger.info(
                f"--- 驗證步驟 {step} --- 平均損失: {avg_loss:.6f}, 平均 SAcc: {avg_sacc:.6f}, 平均 CAcc: {avg_cacc:.6f}"
            )
            writer.add_scalar("Loss/validation", avg_loss, step)
            writer.add_scalar("SAcc/validation", avg_sacc, step)
            writer.add_scalar("CAcc/validation", avg_cacc, step)

            current_score_sum = avg_sacc + avg_cacc

            # 將當前的驗證分數傳給 scheduler
            scheduler.step(current_score_sum)

            if current_score_sum > best_score_sum:
                best_score_sum = current_score_sum

                best_model_path = os.path.join(
                    output_folder, "checkpoints", "best_model.pth"
                )
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": transformer.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "teacher_forcing_ratio": tf_ratio,
                        "best_score_sum": best_score_sum,
                    },
                    best_model_path,
                )
                logger.info(
                    f"*** 新的最佳模型已儲存至 {best_model_path}！ Score Sum: {best_score_sum:.6f} (SAcc: {avg_sacc:.6f}, CAcc: {avg_cacc:.6f}) ***"
                )

            if step % hyper_params["saveModelStep"] == 0 and step > 0:
                ckpt_path = os.path.join(
                    output_folder, "checkpoints", f"model_step_{step}.pth"
                )
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": transformer.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "teacher_forcing_ratio": tf_ratio,
                        "best_score_sum": best_score_sum,
                    },
                    ckpt_path,
                )
                logger.info(f"模型已按步數儲存至: {ckpt_path}")
        step += 1
    writer.close()
    logger.info("--- 訓練完成 ---")


def test_net():
    logger.info("--- 開始測試 ---")
    img_folder = os.path.join(output_folder, "imgs")
    os.makedirs(img_folder, exist_ok=True)
    val_loss, val_sacc, val_cacc, count = 0, 0, 0, 0
    test_iter = iter(test_loader)
    test_itr = 1
    while True:
        try:
            batch_data = next(test_iter)
            loss_v, _, sacc_v, cacc_v, final_predictions = test_autoregre_step(
                batch_data
            )
            val_loss += loss_v.item()
            val_sacc += sacc_v.item()
            val_cacc += cacc_v
            count += 1
            logger.info(
                f"測試樣本 {test_itr}, SAcc: {sacc_v.item():.6f}, CAcc: {cacc_v:.6f}"
            )
            input_raw, glabel_raw, nb_strokes, _ = batch_data
            gt_vis_img = generate_grouped_image(
                input_raw[0, :, :, : nb_strokes[0]], glabel_raw[0]
            )
            pred_vis_img = generate_grouped_image(
                input_raw[0, :, :, : nb_strokes[0]],
                torch.round(torch.sigmoid(final_predictions[0])),
            )

            save_image_safe(
                gt_vis_img, os.path.join(img_folder, f"{test_itr}_g_label.jpeg")
            )
            save_image_safe(
                pred_vis_img, os.path.join(img_folder, f"{test_itr}_g_pred.jpeg")
            )
            test_itr += 1
        except StopIteration:
            break
    avg_loss, avg_sacc, avg_cacc = val_loss / count, val_sacc / count, val_cacc / count
    logger.info(
        f"測試完成 - 平均損失: {avg_loss:.6f}, 平均 SAcc: {avg_sacc:.6f}, 平均 CAcc: {avg_cacc:.6f}"
    )


# ====== 主程式 ======
if __name__ == "__main__":
    args = parser.parse_args()
    hyper_params = vars(args)

    output_folder = os.path.join(
        hyper_params["outDir"], f'_{datetime.datetime.now().strftime(r"%Y%m%d_%H%M%S")}'
    )
    os.makedirs(output_folder, exist_ok=True)

    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_folder, "log.txt"))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用設備: {device}")

    en_modelAESSG = AutoencoderEmbed(
        code_size=hyper_params["d_model"], x_dim=256, y_dim=256, root_feature=32
    ).to(device)
    de_modelAESSG = AutoencoderEmbed(
        code_size=hyper_params["d_model"], x_dim=256, y_dim=256, root_feature=32
    ).to(device)
    transformer = GpTransformer(
        num_layers=hyper_params["num_layers"],
        d_model=hyper_params["d_model"],
        num_heads=hyper_params["num_heads"],
        dff=hyper_params["d_ff"],
        pe_input=512,
        pe_target=64,
        rate=hyper_params["drop_rate"],
    ).to(device)
    optimizer = optim.Adam(
        transformer.parameters(), lr=hyper_params["lr"], betas=(0.9, 0.98), eps=1e-9
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    logger.info("正在準備資料載入器...")
    train_loader = None
    if hyper_params["status"] == "train":
        train_loader = DataLoader(
            GPRegDataset(
                data_dir=hyper_params["dbDir"],
                raw_size=[256, 256],
                prefix="train",
                augment=True,
            ),
            batch_size=hyper_params["batchSize"],
            shuffle=True,
            collate_fn=gpreg_collate_fn,
        )

    test_loader = DataLoader(
        GPRegDataset(
            data_dir=hyper_params["dbDir"], raw_size=[256, 256], prefix="test"
        ),
        batch_size=1,
        shuffle=False,
        collate_fn=gpreg_collate_fn,
    )

    logger.info(f"正在從 {hyper_params['embed_ckpt']} 載入預訓練的 Embedding 模型...")
    try:
        checkpoint_embed = torch.load(hyper_params["embed_ckpt"], map_location=device)
        en_modelAESSG.load_state_dict(checkpoint_embed["model_state_dict"])
        de_modelAESSG.load_state_dict(checkpoint_embed["model_state_dict"])
        en_modelAESSG.eval()
        de_modelAESSG.eval()
        logger.info("Embedding 模型載入成功。")
    except Exception as e:
        logger.error(f"載入 Embedding checkpoint 失敗: {e}")
        exit()

    if hyper_params["ckpt"]:
        logger.info(f"從 checkpoint 載入 Transformer 模型: {hyper_params['ckpt']}")
        checkpoint = torch.load(hyper_params["ckpt"], map_location=device)
        transformer.load_state_dict(checkpoint["model_state_dict"])
        if hyper_params["status"] == "train" and hyper_params["cnt"]:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.info("成功恢復 Scheduler 狀態。")
            hyper_params["start_step"] = checkpoint.get("step", 0)
            # 將 best_score_sum 也恢復，以確保 scheduler 和 best model 儲存邏輯的連續性
            hyper_params["best_score_sum"] = checkpoint.get("best_score_sum", 0.0)
            logger.info(f"成功恢復訓練，從步驟 {hyper_params['start_step']} 開始。")

    if hyper_params["status"] == "train":
        if not train_loader:
            logger.error("訓練模式下，train 資料夾不可為空。")
            exit()
        train_net()
    elif hyper_params["status"] == "test":
        test_net()
