import os
import logging
import argparse
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2

# 從您提供的檔案中匯入模組
from loader import GPRegDataset, gpreg_collate_fn
from network import GpTransformer, AutoencoderEmbed

# ====== 參數解析 ======
parser = argparse.ArgumentParser(description="訓練 Sketch Semantic Segmentation Transformer 模型 (PyTorch, 原版風格)")
# --- 路徑與模式設定 ---
parser.add_argument('--status', help='執行模式 (train 或 test)', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--dbDir', help='由 write_data*.py 產生的 .pt 資料庫目錄', type=str, required=True)
parser.add_argument('--outDir', help='輸出結果的根目錄', type=str, default='result_segformer_pytorch')
parser.add_argument('--embed_ckpt', help='預訓練 Embedding Autoencoder 的 .pth checkpoint 路徑', type=str, required=True)
parser.add_argument('--ckpt', help='Segformer 模型的 .pth checkpoint 路徑 (若要繼續訓練或測試)', type=str, default=None)
parser.add_argument('--cnt', help='是否從 ckpt 繼續訓練', action='store_true')
# --- 訓練超參數 ---
parser.add_argument('--maxIter', help='最大訓練步數 (迭代次數)', type=int, default=200000)
parser.add_argument('--batchSize', help='批次大小', type=int, default=16)
parser.add_argument('--lr', help='學習率', type=float, default=1e-4)
# --- 模型超參數 ---
parser.add_argument('--d_model', help='模型的特徵維度', type=int, default=256)
parser.add_argument('--num_layers', help='Transformer Encoder/Decoder 層數', type=int, default=4)
parser.add_argument('--d_ff', help='前饋網路的中間層維度', type=int, default=2048)
parser.add_argument('--num_heads', help='多頭注意力機制的頭數', type=int, default=4)
parser.add_argument('--drop_rate', help='Dropout 比率', type=float, default=0.4)
# --- 記錄與儲存頻率 ---
parser.add_argument('--dispLossStep', help='每隔多少步顯示一次日誌', type=int, default=100)
parser.add_argument('--exeValStep', help='每隔多少步驗證一次模型', type=int, default=2000)
parser.add_argument('--saveModelStep', help='每隔多少步儲存一次模型', type=int, default=2000)

# ====== 輔助函式 (與上一版相同) ======
def create_masks_pytorch(inp, tar):
    device = inp.device
    enc_padding_mask = (torch.sum(inp, dim=2) == -2.0 * inp.shape[2]).unsqueeze(1).unsqueeze(2)
    dec_padding_mask = enc_padding_mask
    tar_len = tar.shape[1]
    look_ahead_mask = torch.triu(torch.ones((tar_len, tar_len), device=device), diagonal=1).bool()
    tar_padding_mask = (torch.sum(tar, dim=2) == -2.0 * tar.shape[2]).unsqueeze(1).unsqueeze(2)
    combined_mask = tar_padding_mask | look_ahead_mask
    return enc_padding_mask, combined_mask, dec_padding_mask

def loss_fn_pytorch(real, pred):
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    mask = (real != -1.0).float()
    loss = criterion(pred, real) * mask
    nb_elem = torch.sum(mask)
    if nb_elem == 0: return torch.tensor(0.0, device=pred.device), torch.tensor(1.0, device=pred.device)
    loss_val = torch.sum(loss) / nb_elem
    pred_sigmoid_masked = torch.round(torch.sigmoid(pred)) * mask
    real_masked = real * mask
    acc_val = 1.0 - (torch.sum(torch.abs(pred_sigmoid_masked - real_masked)) / nb_elem)
    return loss_val, acc_val

def cacc_pytorch(label_pred_sigmoid, gt_label):
    label_pred = torch.round(label_pred_sigmoid[0]).detach().cpu().numpy()
    gt_label = gt_label[0].detach().cpu().numpy()
    valid_strokes_mask = gt_label[0, :] != -1.0
    if not np.any(valid_strokes_mask): return 0.0
    gt_label = gt_label[:, valid_strokes_mask]
    label_pred = label_pred[:, valid_strokes_mask]
    valid_groups_mask = np.sum(gt_label, axis=1) != -1.0 * gt_label.shape[1]
    if not np.any(valid_groups_mask): return 0.0
    gt_label = gt_label[valid_groups_mask, :]
    label_pred = label_pred[valid_groups_mask, :]
    if gt_label.shape[1] == 0: return 0.0
    category_list = np.argmax(gt_label, axis=0)
    prediction_list = np.argmax(label_pred, axis=0)
    group_ids, group_counts = np.unique(category_list, return_counts=True)
    correct_groups = 0
    for gid, total in zip(group_ids, group_counts):
        correct_mask = (category_list == gid) & (prediction_list == gid)
        correct_count = np.sum(correct_mask)
        if (correct_count / total) >= 0.75: correct_groups += 1
    return correct_groups / len(group_ids) if len(group_ids) > 0 else 0.0

def group_images_pytorch(images, labels):
    images = images.permute(2, 0, 1)
    grouped_images = []
    num_groups = labels.shape[0]
    for i in range(num_groups):
        indices = torch.where(labels[i] == 1)[0]
        if len(indices) > 0:
            grouped_images.append(torch.sum(torch.index_select(images, 0, indices), dim=0))
        else:
            grouped_images.append(torch.zeros_like(images[0]))
    return torch.clamp(torch.stack(grouped_images), 0, 1)

def cook_raw_pytorch(batch_data):
    # 隱含地使用全域變數 encoder_net 和 device
    input_raw, glabel_raw, nb_strokes, nb_gps = batch_data
    input_raw, glabel_raw = input_raw.to(device), glabel_raw.to(device)
    batch_size = input_raw.shape[0]
    input_embeddings, target_embeddings = [], []
    for i in range(batch_size):
        single_input_strokes = input_raw[i, :, :, :nb_strokes[i]].permute(2, 0, 1).unsqueeze(1)
        single_labels = glabel_raw[i, :nb_gps[i], :nb_strokes[i]]
        stroke_embeds = encoder_net.encode(single_input_strokes)
        input_embeddings.append(stroke_embeds)
        grouped_imgs = group_images_pytorch(input_raw[i, :, :, :nb_strokes[i]], single_labels)
        group_embeds = encoder_net.encode(grouped_imgs.unsqueeze(1))
        start_token = torch.full((1, group_embeds.shape[1]), -1.0, device=device)
        target_embeds_with_start = torch.cat([start_token, group_embeds], dim=0)
        target_embeddings.append(target_embeds_with_start)
    padded_input_embeds = nn.utils.rnn.pad_sequence(input_embeddings, batch_first=True, padding_value=-2.0)
    padded_target_embeds = nn.utils.rnn.pad_sequence(target_embeddings, batch_first=True, padding_value=-2.0)
    return padded_input_embeds, padded_target_embeds, glabel_raw

# ====== 訓練與測試流程 (函式不帶參數，隱含地使用全域變數) ======
def train_step(batch_data):
    transformer.train()
    inp_embed, tar_embed, labels = cook_raw_pytorch(batch_data)
    tar_for_input = tar_embed[:, :-1, :]
    tar_for_loss = labels
    enc_mask, combined_mask, dec_mask = create_masks_pytorch(inp_embed, tar_for_input)
    optimizer.zero_grad()
    predictions, _ = transformer(inp_embed, tar_for_input, enc_mask, combined_mask, dec_mask)
    loss, acc = loss_fn_pytorch(tar_for_loss, predictions)
    loss.backward()
    optimizer.step()
    return loss, acc

def test_step(batch_data):
    transformer.eval()
    with torch.no_grad():
        input_raw, glabel_raw, nb_strokes, nb_gps = batch_data
        input_raw, glabel_raw = input_raw.to(device), glabel_raw.to(device)
        single_input_strokes = input_raw[0, :, :, :nb_strokes[0]].permute(2, 0, 1).unsqueeze(1)
        inp_embed = encoder_net.encode(single_input_strokes).unsqueeze(0)
        gp_token = torch.full((1, 1, hyper_params['d_model']), -1.0, device=device)
        num_groups_to_predict = glabel_raw.shape[1]
        all_predictions = []
        for _ in range(num_groups_to_predict):
            enc_mask, combined_mask, dec_mask = create_masks_pytorch(inp_embed, gp_token)
            predictions, _ = transformer(inp_embed, gp_token, enc_mask, combined_mask, dec_mask)
            last_pred = predictions[:, -1:, :]
            all_predictions.append(last_pred)
            pred_labels_for_group = torch.round(torch.sigmoid(last_pred.squeeze(1)))
            grouped_img = group_images_pytorch(input_raw[0, :, :, :nb_strokes[0]], pred_labels_for_group)
            new_token_embed = encoder_net.encode(grouped_img.unsqueeze(1))
            gp_token = torch.cat([gp_token, new_token_embed.unsqueeze(1)], dim=1)
        final_predictions = torch.cat(all_predictions, dim=1)
        loss, acc = loss_fn_pytorch(glabel_raw, final_predictions)
        cacc = cacc_pytorch(torch.sigmoid(final_predictions), glabel_raw)
    return loss, acc, cacc

def train_net():
    train_logger = logging.getLogger('main.training')
    train_logger.info('---Begin training: ---')
    writer = SummaryWriter(log_dir=os.path.join(output_folder, 'summary'))
    step = hyper_params.get('start_step', 0)
    train_iter = iter(train_loader)

    while step < hyper_params['maxIter']:
        try:
            batch_data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch_data = next(train_iter)

        loss, acc = train_step(batch_data)

        if step % hyper_params['dispLossStep'] == 0:
            train_logger.info(f"步驟 [{step}/{hyper_params['maxIter']}], 訓練損失: {loss.item():.4f}, 訓練準確率: {acc.item():.4f}")
            writer.add_scalar('Loss/train', loss.item(), step)
            writer.add_scalar('Accuracy/train', acc.item(), step)

        if step > 0 and step % hyper_params['exeValStep'] == 0:
            total_loss, total_acc, total_cacc, count = 0, 0, 0, 0
            test_iter = iter(test_loader)
            while True:
                try:
                    batch_data = next(test_iter)
                    loss_val, acc_val, cacc_val = test_step(batch_data)
                    total_loss += loss_val.item()
                    total_acc += acc_val.item()
                    total_cacc += cacc_val
                    count += 1
                except StopIteration:
                    break
            avg_loss, avg_acc, avg_cacc = total_loss/count, total_acc/count, total_cacc/count
            train_logger.info(f"驗證完成 - 平均損失: {avg_loss:.4f}, 平均準確率: {avg_acc:.4f}, CAcc: {avg_cacc:.4f}")
            writer.add_scalar('Loss/validation', avg_loss, step)
            writer.add_scalar('Accuracy/validation', avg_acc, step)
            writer.add_scalar('C-Accuracy/validation', avg_cacc, step)

            if step % hyper_params['saveModelStep'] == 0:
                ckpt_path = os.path.join(output_folder, 'checkpoints', f'model_step_{step}.pth')
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save({'step': step, 'model_state_dict': transformer.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, ckpt_path)
                train_logger.info(f"模型已儲存至: {ckpt_path}")
        step += 1
    writer.close()
    train_logger.info("--- 訓練完成 ---")

def test_net():
    test_logger = logging.getLogger('main.testing')
    test_logger.info('---Begin testing: ---')
    total_loss, total_acc, total_cacc, count = 0, 0, 0, 0
    test_iter = iter(test_loader)
    while True:
        try:
            batch_data = next(test_iter)
            loss_val, acc_val, cacc_val = test_step(batch_data)
            total_loss += loss_val.item()
            total_acc += acc_val.item()
            total_cacc += cacc_val
            count += 1
        except StopIteration:
            break
    avg_loss, avg_acc, avg_cacc = total_loss/count, total_acc/count, total_cacc/count
    test_logger.info(f"測試完成 - 平均損失: {avg_loss:.4f}, 平均準確率: {avg_acc:.4f}, CAcc: {avg_cacc:.4f}")

# ====== 主程式 ======
if __name__ == '__main__':
    args = parser.parse_args()
    hyper_params = vars(args)

    output_folder = os.path.join(hyper_params['outDir'], f'_{datetime.datetime.now().strftime(r"%Y%m%d_%H%M%S")}')
    os.makedirs(output_folder, exist_ok=True)
    
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_folder, 'log.txt'))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用設備: {device}")

    # --- 全域變數定義 ---
    encoder_net = AutoencoderEmbed(code_size=hyper_params['d_model'], x_dim=256, y_dim=256, root_feature=32).to(device)
    transformer = GpTransformer(num_layers=hyper_params['num_layers'], d_model=hyper_params['d_model'],
                                num_heads=hyper_params['num_heads'], dff=hyper_params['d_ff'],
                                pe_input=512, pe_target=64, rate=hyper_params['drop_rate']).to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=hyper_params['lr'], betas=(0.9, 0.98), eps=1e-9)
    
    # --- 載入資料 ---
    logger.info("正在準備資料載入器...")
    train_loader = DataLoader(GPRegDataset(data_dir=hyper_params['dbDir'], raw_size=[256, 256], prefix='train'), 
                              batch_size=hyper_params['batchSize'], shuffle=True, collate_fn=gpreg_collate_fn)
    test_loader = DataLoader(GPRegDataset(data_dir=hyper_params['dbDir'], raw_size=[256, 256], prefix='test'), 
                             batch_size=1, shuffle=False, collate_fn=gpreg_collate_fn)

    # --- 載入 Checkpoints ---
    logger.info("正在載入預訓練的 Embedding Autoencoder...")
    try:
        encoder_net.load_state_dict(torch.load(hyper_params['embed_ckpt'], map_location=device)['model_state_dict'])
        encoder_net.eval()
        logger.info("Embedding Autoencoder 載入成功。")
    except Exception as e:
        logger.error(f"載入 Embedding checkpoint 失敗: {e}")
        exit()

    if hyper_params['ckpt']:
        logger.info(f"從 checkpoint 載入 Transformer 模型: {hyper_params['ckpt']}")
        checkpoint = torch.load(hyper_params['ckpt'], map_location=device)
        transformer.load_state_dict(checkpoint['model_state_dict'])
        if hyper_params['status'] == 'train' and hyper_params['cnt']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            hyper_params['start_step'] = checkpoint.get('step', 0)
            logger.info(f"成功恢復訓練，從步驟 {hyper_params['start_step']} 開始。")

    # --- 執行 ---
    if hyper_params['status'] == 'train':
        train_net()
    elif hyper_params['status'] == 'test':
        test_net()