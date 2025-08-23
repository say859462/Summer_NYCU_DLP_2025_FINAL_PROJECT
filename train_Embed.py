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
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
from PIL import Image
import numpy as np

# 假設 loader.py 和 network.py 都在同一個目錄或 Python 路徑中
from loader import AeDataset
from network import AutoencoderEmbed

# Hyper Parameters (基於論文設定)
hyper_params = {
    "maxIter": 1500000,
    "batchSize": 64,
    "dbDir": "data_embed_pt",
    "outDir": "result_embed",
    "device": "0",  # 會在 main block 中被 torch 的 device 取代
    "rootFt": 32,
    "dispLossStep": 200,
    "exeValStep": 5000,
    "saveModelStep": 5000,
    "nbDispImg": 4,
    "ckpt": None,
    "cnt": False,
    "status": "train",  # 預設狀態
    "codeSize": 256,
    "imgSize": 256,
    "gamma": 0.8,
    "lr": 1e-4,  # 論文指定的學習率
}


# =============================================================================
# 核心損失函數 (嚴格遵循論文)
# =============================================================================
# def loss_fn_for_train(recon_pred, labels, dist_pred, dist_labels, gamma):
#     """計算訓練時的總損失 (參照論文 Eq. 2, 3, 4)"""
#     recon_loss = nn.MSELoss()(recon_pred, labels)
#     dist_loss = nn.MSELoss()(dist_pred, dist_labels)
#     loss = recon_loss + gamma * dist_loss
#     return loss, recon_loss, dist_loss


def loss_fn_for_train(recon_pred, labels, dist_pred, dist_labels, alpha=0.8):
    # Sigmoid 交叉熵損失 (類似 tf.nn.sigmoid_cross_entropy_with_logits)
    cons_loss = nn.BCEWithLogitsLoss()(recon_pred, labels)
    pre_loss = nn.MSELoss()(dist_pred, dist_labels)
    loss = alpha * cons_loss + (1 - alpha) * pre_loss
    return loss, cons_loss, pre_loss


def loss_fn_for_eval(recon_pred, labels):
    loss = nn.BCEWithLogitsLoss()(recon_pred, labels)
    preds = torch.sigmoid(recon_pred) > 0.5
    acc = (preds == (labels > 0.5)).float().mean().item()
    return loss.item(), acc


# =============================================================================


def train_net(logger, output_folder, device):
    """訓練網路的主函式"""
    train_logger = logging.getLogger("main.training")
    train_logger.info("---Begin training: ---")

    model = AutoencoderEmbed(
        hyper_params["codeSize"],
        hyper_params["imgSize"],
        hyper_params["imgSize"],
        hyper_params["rootFt"],
    ).to(device)

    try:
        train_dataset = AeDataset(
            hyper_params["dbDir"],
            [hyper_params["imgSize"], hyper_params["imgSize"]],
            "train",
        )
        valid_dataset = AeDataset(
            hyper_params["dbDir"],
            [hyper_params["imgSize"], hyper_params["imgSize"]],
            "valid",
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=hyper_params["batchSize"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=hyper_params["batchSize"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    except FileNotFoundError as e:
        train_logger.error(f"資料載入失敗: {e}")
        return

    optimizer = optim.Adam(model.parameters(), lr=hyper_params["lr"])
    writer = SummaryWriter(log_dir=os.path.join(output_folder, "summary"))
    ckpt_dir = os.path.join(output_folder, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    start_step = 0
    if (
        hyper_params["cnt"]
        and hyper_params["ckpt"]
        and os.path.exists(hyper_params["ckpt"])
    ):
        train_logger.info(f"從 {hyper_params['ckpt']} 恢復訓練...")
        checkpoint = torch.load(hyper_params["ckpt"])
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint.get("step", 0)
        train_logger.info(f"成功載入權重，從步驟 {start_step} 繼續。")

    step = start_step
    train_iter = iter(train_loader)

    while step < hyper_params["maxIter"]:
        try:
            inputs, dist_labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, dist_labels = next(train_iter)

        model.train()
        inputs = inputs.to(device)
        dist_labels = dist_labels.to(device)

        recon_logits, dist_logits = model(inputs)
        recon_pred = torch.sigmoid(recon_logits)
        dist_pred = torch.sigmoid(dist_logits)

        loss, cons_loss, pre_loss = loss_fn_for_train(
            recon_pred, inputs, dist_pred, dist_labels, hyper_params["gamma"]
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % hyper_params["dispLossStep"] == 0:
            train_logger.info(
                f"步驟 [{step}/{hyper_params['maxIter']}], 總損失: {loss.item():.6f}, "
                f"重建損失: {cons_loss.item():.6f}, 距離場損失: {pre_loss.item():.6f}"
            )
            writer.add_scalar("Loss/train_total", loss.item(), step)
            writer.add_scalar("Loss/train_reconstruction", cons_loss.item(), step)
            writer.add_scalar("Loss/train_distance_field", pre_loss.item(), step)

        if step > 0 and step % hyper_params["exeValStep"] == 0:
            model.eval()
            total_val_loss, total_val_acc = 0.0, 0.0
            with torch.no_grad():
                for i, (val_inputs, _) in enumerate(valid_loader):
                    val_inputs = val_inputs.to(device)
                    val_recon_logits, _ = model(val_inputs)
                    val_recon_pred = torch.sigmoid(val_recon_logits)
                    val_loss, val_acc = loss_fn_for_eval(val_recon_pred, val_inputs)
                    total_val_loss += val_loss
                    total_val_acc += val_acc
                    if i == 0:
                        grid = make_grid(
                            torch.cat([val_inputs[:8].cpu(), val_recon_pred[:8].cpu()]),
                            nrow=8,
                        )
                        writer.add_image("Validation/reconstruction", grid, step)

            avg_val_loss = total_val_loss / len(valid_loader)
            avg_val_acc = total_val_acc / len(valid_loader)
            writer.add_scalar("Loss/validation", avg_val_loss, step)
            writer.add_scalar("Accuracy/validation", avg_val_acc, step)
            train_logger.info(
                f"--- 驗證步驟 {step} --- 平均損失: {avg_val_loss:.6f}, 平均準確率: {avg_val_acc:.6f}"
            )

        if step > 0 and step % hyper_params["saveModelStep"] == 0:
            ckpt_path = os.path.join(ckpt_dir, f"model_step_{step}.pth")
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )
            train_logger.info(f"模型已儲存至: {ckpt_path}")

        step += 1
    writer.close()
    train_logger.info("--- 訓練完成 ---")


# =============================================================================
# 測試與分析函式
# =============================================================================


def test_net(logger, output_folder, device):
    """測試網路並輸出最終評估指標"""
    test_logger = logging.getLogger("main.testing")
    test_logger.info("---Begin testing: ---")

    model = AutoencoderEmbed(
        hyper_params["codeSize"],
        hyper_params["imgSize"],
        hyper_params["imgSize"],
        hyper_params["rootFt"],
    ).to(device)

    if not hyper_params["ckpt"] or not os.path.exists(hyper_params["ckpt"]):
        test_logger.error(
            f"必須提供有效的權重檔案路徑 (--ckpt)。找不到: {hyper_params['ckpt']}"
        )
        return

    test_logger.info(f"從 {hyper_params['ckpt']} 載入權重...")
    checkpoint = torch.load(hyper_params["ckpt"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    try:
        test_dataset = AeDataset(
            hyper_params["dbDir"],
            [hyper_params["imgSize"], hyper_params["imgSize"]],
            "test",
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=hyper_params["batchSize"],
            shuffle=False,
            num_workers=4,
        )
    except FileNotFoundError as e:
        test_logger.error(f"資料載入失敗: {e}")
        return

    total_loss, total_acc = 0.0, 0.0
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            recon_logits, _ = model(inputs)
            recon_pred = torch.sigmoid(recon_logits)
            loss, acc = loss_fn_for_eval(recon_pred, inputs)
            total_loss += loss.item() * inputs.size(0)
            total_acc += acc * inputs.size(0)

    avg_loss = total_loss / len(test_dataset)
    avg_acc = total_acc / len(test_dataset)
    test_logger.info(f"測試完成 -> 平均損失: {avg_loss:.6f}, 平均準確率: {avg_acc:.6f}")


def output_vis(logger, output_folder, device):
    """視覺化輸出，儲存原始圖與重建圖 (黑底白線)"""
    vis_logger = logging.getLogger("main.visualizing")
    vis_logger.info("---Begin visualizing: ---")

    model = AutoencoderEmbed(
        hyper_params["codeSize"],
        hyper_params["imgSize"],
        hyper_params["imgSize"],
        hyper_params["rootFt"],
    ).to(device)

    if not hyper_params["ckpt"] or not os.path.exists(hyper_params["ckpt"]):
        vis_logger.error(
            f"必須提供有效的權重檔案路徑 (--ckpt)。找不到: {hyper_params['ckpt']}"
        )
        return

    vis_logger.info(f"從 {hyper_params['ckpt']} 載入權重...")
    checkpoint = torch.load(hyper_params["ckpt"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    try:
        vis_dataset = AeDataset(
            hyper_params["dbDir"],
            [hyper_params["imgSize"], hyper_params["imgSize"]],
            "test",
        )
        vis_loader = DataLoader(vis_dataset, batch_size=1, shuffle=False)
    except FileNotFoundError as e:
        vis_logger.error(f"資料載入失敗: {e}")
        return

    img_folder = os.path.join(output_folder, "imgs_vis")
    os.makedirs(img_folder, exist_ok=True)

    with torch.no_grad():
        for i, (inputs, _) in enumerate(vis_loader):
            if i >= 3000:
                break
            inputs = inputs.to(device)
            recon_logits, _ = model(inputs)
            recon_imgs = torch.sigmoid(recon_logits)

            save_image(inputs, os.path.join(img_folder, f"{i}_original.jpeg"))
            save_image(recon_imgs, os.path.join(img_folder, f"{i}_reconstructed.jpeg"))
            if (i + 1) % 100 == 0:
                vis_logger.info(f"已儲存 {i+1} 張圖像...")

    vis_logger.info(f"視覺化圖像已儲存至: {img_folder}")


def test_code(logger, output_folder, device):
    """兩點插值測試，模擬原始碼的 codeItp 模式"""
    interp_logger = logging.getLogger("main.interpolating")
    interp_logger.info("---Begin interpolating: ---")

    model = AutoencoderEmbed(
        hyper_params["codeSize"],
        hyper_params["imgSize"],
        hyper_params["imgSize"],
        hyper_params["rootFt"],
    ).to(device)

    if not hyper_params["ckpt"] or not os.path.exists(hyper_params["ckpt"]):
        interp_logger.error(
            f"必須提供有效的權重檔案路徑 (--ckpt)。找不到: {hyper_params['ckpt']}"
        )
        return

    interp_logger.info(f"從 {hyper_params['ckpt']} 載入權重...")
    checkpoint = torch.load(hyper_params["ckpt"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    try:
        interp_dataset = AeDataset(
            hyper_params["dbDir"],
            [hyper_params["imgSize"], hyper_params["imgSize"]],
            "test",
        )
        interp_loader = DataLoader(interp_dataset, batch_size=2, shuffle=True)
    except FileNotFoundError as e:
        interp_logger.error(f"資料載入失敗: {e}")
        return

    img_folder = os.path.join(output_folder, "imgs_codeItp")
    os.makedirs(img_folder, exist_ok=True)

    with torch.no_grad():
        for i, (inputs, _) in enumerate(interp_loader):
            if i >= 500:
                break
            inputs = inputs.to(device)
            codes = model.encoder(inputs)

            recons = model.decoder_for_cons(codes).sigmoid()
            save_image(inputs[0], os.path.join(img_folder, f"{i}_original_A.jpeg"))
            save_image(inputs[1], os.path.join(img_folder, f"{i}_original_B.jpeg"))

            for j, ratio in enumerate(torch.linspace(0, 1, 11, device=device)):
                interp_code = torch.lerp(codes[0], codes[1], ratio).unsqueeze(0)
                interp_image = model.decoder_for_cons(interp_code).sigmoid()
                save_image(
                    interp_image, os.path.join(img_folder, f"{i}_interp_{j:02d}.jpeg")
                )

            if (i + 1) % 20 == 0:
                interp_logger.info(f"已生成 {i+1} 組插值影像...")
    interp_logger.info("插值影像生成完畢!")


def test_code3(logger, output_folder, device):
    """三點插值測試，模擬原始碼的 tripleItp 模式"""
    interp_logger = logging.getLogger("main.interpolating3")
    interp_logger.info("---Begin triple interpolating: ---")

    model = AutoencoderEmbed(
        hyper_params["codeSize"],
        hyper_params["imgSize"],
        hyper_params["imgSize"],
        hyper_params["rootFt"],
    ).to(device)

    if not hyper_params["ckpt"] or not os.path.exists(hyper_params["ckpt"]):
        interp_logger.error(
            f"必須提供有效的權重檔案路徑 (--ckpt)。找不到: {hyper_params['ckpt']}"
        )
        return

    interp_logger.info(f"從 {hyper_params['ckpt']} 載入權重...")
    checkpoint = torch.load(hyper_params["ckpt"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    try:
        interp_dataset = AeDataset(
            hyper_params["dbDir"],
            [hyper_params["imgSize"], hyper_params["imgSize"]],
            "test",
        )
        interp_loader = DataLoader(interp_dataset, batch_size=3, shuffle=True)
    except FileNotFoundError as e:
        interp_logger.error(f"資料載入失敗: {e}")
        return

    img_folder = os.path.join(output_folder, "imgs_tripleItp")
    os.makedirs(img_folder, exist_ok=True)

    with torch.no_grad():
        for i, (inputs, _) in enumerate(interp_loader):
            if i >= 50:
                break
            inputs = inputs.to(device)
            codes = model.encoder(inputs)

            avg_code = codes.mean(dim=0, keepdim=True)
            avg_image = model.decoder_for_cons(avg_code).sigmoid()

            save_image(inputs[0], os.path.join(img_folder, f"{i}_original_A.jpeg"))
            save_image(inputs[1], os.path.join(img_folder, f"{i}_original_B.jpeg"))
            save_image(inputs[2], os.path.join(img_folder, f"{i}_original_C.jpeg"))
            save_image(avg_image, os.path.join(img_folder, f"{i}_average.jpeg"))

            if (i + 1) % 10 == 0:
                interp_logger.info(f"已生成 {i+1} 組三點平均影像...")
    interp_logger.info("三點平均影像生成完畢!")


def test_white_image(logger, output_folder, device):
    """生成白底黑線的視覺化結果"""
    vis_logger = logging.getLogger("main.white_image")
    vis_logger.info("---Begin white image visualization: ---")

    model = AutoencoderEmbed(
        hyper_params["codeSize"],
        hyper_params["imgSize"],
        hyper_params["imgSize"],
        hyper_params["rootFt"],
    ).to(device)

    if not hyper_params["ckpt"] or not os.path.exists(hyper_params["ckpt"]):
        vis_logger.error(
            f"必須提供有效的權重檔案路徑 (--ckpt)。找不到: {hyper_params['ckpt']}"
        )
        return

    vis_logger.info(f"從 {hyper_params['ckpt']} 載入權重...")
    checkpoint = torch.load(hyper_params["ckpt"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    try:
        vis_dataset = AeDataset(
            hyper_params["dbDir"],
            [hyper_params["imgSize"], hyper_params["imgSize"]],
            "test",
        )
        vis_loader = DataLoader(
            vis_dataset, batch_size=hyper_params["batchSize"], shuffle=False
        )
    except FileNotFoundError as e:
        vis_logger.error(f"資料載入失敗: {e}")
        return

    img_folder = os.path.join(output_folder, "imgs_white")
    os.makedirs(img_folder, exist_ok=True)

    count = 0
    with torch.no_grad():
        for inputs, _ in vis_loader:
            if count >= 3000:
                break
            inputs = inputs.to(device)
            recons = model.decoder_for_cons(model.encoder(inputs)).sigmoid()

            for j in range(inputs.size(0)):
                if count >= 3000:
                    break

                original_img = 1 - inputs[j].cpu().numpy().squeeze()
                recon_img = 1 - recons[j].cpu().numpy().squeeze()

                Image.fromarray((original_img * 255).astype(np.uint8)).save(
                    os.path.join(img_folder, f"{count}_original.jpeg")
                )
                Image.fromarray((recon_img * 255).astype(np.uint8)).save(
                    os.path.join(img_folder, f"{count}_reconstructed.jpeg")
                )
                count += 1

            vis_logger.info(f"已儲存 {count} 張圖像...")

    vis_logger.info("白底圖像生成完畢!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", help="GPU device indices", type=str, default="0")
    parser.add_argument("--ckpt", help="checkpoint path", type=str, default=None)
    parser.add_argument("--cnt", help="continue training flag", action="store_true")
    parser.add_argument("--rootFt", help="root feature size", type=int, default=32)
    parser.add_argument(
        "--status",
        help="執行模式",
        type=str,
        default="train",
        choices=["train", "test", "vis", "codeItp", "tripleItp", "white_image"],
    )
    parser.add_argument(
        "--dbDir", help="database directory", type=str, default="data_embed_pt"
    )
    parser.add_argument("--outDir", help="output directory", type=str, default="result")
    args = parser.parse_args()

    # 更新超參數
    hyper_params["device"] = (
        f"cuda:{args.devices.split(',')[0]}" if torch.cuda.is_available() else "cpu"
    )
    hyper_params["ckpt"] = args.ckpt
    hyper_params["cnt"] = args.cnt
    hyper_params["rootFt"] = args.rootFt
    hyper_params["status"] = args.status
    hyper_params["dbDir"] = args.dbDir
    hyper_params["outDir"] = args.outDir

    timeSufix = time.strftime(r"%Y%m%d_%H%M%S")
    output_folder = os.path.join(hyper_params["outDir"], f"_{timeSufix}")
    os.makedirs(output_folder, exist_ok=True)

    # 設定 Logger
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_folder, "log.txt"))
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    device = torch.device(hyper_params["device"])

    # 根據狀態開始執行
    if hyper_params["status"] == "train":
        train_net(logger, output_folder, device)
    elif hyper_params["status"] == "test":
        test_net(logger, output_folder, device)
    elif hyper_params["status"] == "vis":
        output_vis(logger, output_folder, device)
    elif hyper_params["status"] == "codeItp":
        test_code(logger, output_folder, device)
    elif hyper_params["status"] == "tripleItp":
        test_code3(logger, output_folder, device)
    elif hyper_params["status"] == "white_image":
        test_white_image(logger, output_folder, device)
    else:
        logger.error(f"未知的狀態: {hyper_params['status']}.")
