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

# 從您提供的 PyTorch 檔案中匯入模組
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
    default=100000,
)
parser.add_argument(
    "--dispLossStep", help="每隔多少步顯示一次日誌", type=int, default=100
)
parser.add_argument(
    "--exeValStep", help="每隔多少步驗證一次模型", type=int, default=2000
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
    canvas = torch.ones(stroke_images.shape[0], stroke_images.shape[1], 3)
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
        group_img = group_images(stroke_images, group_labels[i : i + 1, :])
        mask = (1.0 - group_img).unsqueeze(-1)
        colored_layer = color.view(1, 1, 3) * mask
        canvas = canvas * (1.0 - mask) + colored_layer
    return (canvas * 255).numpy().astype(np.uint8)


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
    step = hyper_params.get("start_step", 0)
    accumulation_steps = hyper_params.get("accum_steps", 1)

    # Teacher forcing 參數
    tf_start = hyper_params.get("teacher_forcing_start", 1.0)
    tf_end = hyper_params.get("teacher_forcing_end", 0.2)
    tf_decay_steps = hyper_params.get("teacher_forcing_decay_steps", 100000)

    best_sacc = 0.0
    best_cacc = 0.0
    best_score_sum = 0.0

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

        # 獲取當前 teacher forcing ratio
        tf_ratio = get_teacher_forcing_ratio(step, tf_start, tf_end, tf_decay_steps)

        # 決定是否使用 teacher forcing
        use_teacher_forcing = True
        if tf_ratio < 1.0:  # 只有在需要混合時才進行
            use_teacher_forcing = torch.rand(1).item() < tf_ratio

        if use_teacher_forcing:
            # 使用 ground truth 作為 decoder 輸入
            tar_for_input = tar_embed[:, :-1, :]
        else:
            # 使用模型預測作為 decoder 輸入 (auto-regressive)
            # 這裡需要模擬推理時的 auto-regressive 過程
            batch_size = inp_embed.shape[0]
            max_groups = tar_embed.shape[1] - 1  # 減去 start token

            # 初始化 decoder 輸入 (只有 start token)
            decoder_input = tar_embed[:, :1, :]  # start token

            # 逐步生成 group tokens
            for group_idx in range(max_groups):
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
                # 這裡需要根據 pred_labels 生成對應的 group 影像並編碼
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

                    # 根據【裁剪後】的預測標籤生成 group 影像
                    grouped_img = group_images(stroke_images, valid_pred_labels)
                    # 編碼 group 影像
                    group_embed = de_modelAESSG.encode(grouped_img.unsqueeze(1))
                    group_embeddings.append(group_embed)

                # 拼接 group embeddings
                new_tokens = torch.stack(group_embeddings, dim=0)
                decoder_input = torch.cat([decoder_input, new_tokens], dim=1)

            tar_for_input = decoder_input

        enc_mask, combined_mask, dec_mask = create_masks(inp_embed, tar_for_input)

        predictions, _ = transformer(
            inp_embed, tar_for_input, enc_mask, combined_mask, dec_mask
        )
        # 在自迴歸模式下，預測序列會比標籤序列多一個，我們需要將其裁切掉
        if predictions.shape[1] > labels.shape[1]:
            predictions = predictions[:, : labels.shape[1], :]
        # 使用 Focal Loss (gamma=2.0)
        loss, acc = loss_fn(labels, predictions)

        loss = loss / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if step % hyper_params["dispLossStep"] == 0:
            sacc_val = sacc(torch.sigmoid(predictions), labels)
            cacc_val = cacc(torch.sigmoid(predictions), labels)
            logger.info(
                f"步驟 [{step}/{hyper_params['maxIter']}], "
                f"損失: {loss.item() * accumulation_steps:.6f}, "
                f"Acc: {acc.item():.6f}, "
                f"SAcc: {sacc_val.item():.6f}, "
                f"CAcc: {cacc_val:.6f}, "
                f"TF Ratio: {tf_ratio:.3f}"
            )
            writer.add_scalar("Loss/train", loss.item() * accumulation_steps, step)
            writer.add_scalar("SAcc/train", sacc_val.item(), step)
            writer.add_scalar("CAcc/train", cacc_val, step)
            writer.add_scalar("Teacher_Forcing_Ratio", tf_ratio, step)

        if step > 0 and step % hyper_params["exeValStep"] == 0:
            val_loss, val_sacc, val_cacc, count = 0, 0, 0, 0
            test_iter = iter(test_loader)
            while True:
                try:
                    batch_data_val = next(test_iter)
                    loss_v, _, sacc_v, cacc_v, _ = test_autoregre_step(batch_data_val)
                    val_loss += loss_v.item()
                    val_sacc += sacc_v.item()
                    val_cacc += cacc_v
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
            if current_score_sum > best_score_sum:
                best_score_sum = current_score_sum
                best_sacc = avg_sacc
                best_cacc = avg_cacc

                best_model_path = os.path.join(
                    output_folder, "checkpoints", "best_model.pth"
                )
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": transformer.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "sacc": best_sacc,
                        "cacc": best_cacc,
                    },
                    best_model_path,
                )
                logger.info(
                    f"*** 新的最佳模型已儲存至 {best_model_path}！ SAcc: {best_sacc:.6f}, CAcc: {best_cacc:.6f} ***"
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
                        "teacher_forcing_ratio": tf_ratio,
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
            cv2.imwrite(
                os.path.join(img_folder, f"{test_itr}_g_label.jpeg"), gt_vis_img
            )
            cv2.imwrite(
                os.path.join(img_folder, f"{test_itr}_g_pred.jpeg"), pred_vis_img
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
            hyper_params["start_step"] = checkpoint.get("step", 0)
            logger.info(f"成功恢復訓練，從步驟 {hyper_params['start_step']} 開始。")

    if hyper_params["status"] == "train":
        if not train_loader:
            logger.error("訓練模式下，train 資料夾不可為空。")
            exit()
        train_net()
    elif hyper_params["status"] == "test":
        test_net()
