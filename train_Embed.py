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

# 假設 loader.py 和 network.py 都在同一個目錄或 Python 路徑中
from loader import AeDataset
from network import AutoencoderEmbed

# Hyper Parameters (與原始 TF 版本一致)
hyper_params = {
    "maxIter": 1500000,
    "batchSize": 64,
    "dbDir": "data",
    "outDir": "result",
    "device": "0",  # 會在 main block 中被 torch 的 device 取代
    "rootFt": 32,
    "dispLossStep": 200,
    "exeValStep": 5000,
    "saveModelStep": 5000,
    "nbDispImg": 4,
    "ckpt": "",
    "cnt": False,
    "status": "train",  # 預設狀態
    "codeSize": 256,
    "imgSize": 256,
    "alpha": 0.8,
    "lr": 1e-4,  # PyTorch 版本額外需要的學習率參數
}


def loss_fn_for_eval(reconstruction_logits, labels):
    """計算評估時的損失和準確率"""
    loss = nn.BCEWithLogitsLoss()(reconstruction_logits, labels)

    # 計算準確率
    preds = torch.sigmoid(reconstruction_logits) > 0.5
    correct = (preds == labels.byte()).sum().item()
    acc = correct / labels.numel()

    return loss, acc


def loss_fn_for_train(recon_logits, labels, dist_pred, dist_labels, alpha=0.8):
    """計算訓練時的總損失"""
    cons_loss = nn.BCEWithLogitsLoss()(recon_logits, labels)
    pre_loss = nn.MSELoss()(dist_pred, dist_labels)
    loss = alpha * cons_loss + (1 - alpha) * pre_loss
    return loss, cons_loss, pre_loss


def train_net(logger, output_folder):
    """訓練網路的主函式"""
    train_logger = logging.getLogger("main.training")
    train_logger.info("---Begin training: ---")

    device = torch.device(hyper_params["device"])
    train_logger.info(f"使用 Device : {device}")
    # define model
    model = AutoencoderEmbed(
        hyper_params["codeSize"],
        hyper_params["imgSize"],
        hyper_params["imgSize"],
        hyper_params["rootFt"],
    ).to(device)

    # define reader
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
        train_logger.error(
            f"請確認 '{hyper_params['dbDir']}' 資料夾中是否存在.pt檔案。"
        )
        return

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=hyper_params["lr"])

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(output_folder, "summary"))

    # Checkpoint
    ckpt_dir = os.path.join(output_folder, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    start_step = 0
    if hyper_params["cnt"] and hyper_params["ckpt"]:
        if os.path.exists(hyper_params["ckpt"]):
            train_logger.info(f"從 {hyper_params['ckpt']} 恢復訓練...")
            checkpoint = torch.load(hyper_params["ckpt"])
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_step = checkpoint["step"]
            train_logger.info(f"成功載入權重，從步驟 {start_step} 繼續。")
        else:
            train_logger.warning(
                f"找不到指定的權重檔案: {hyper_params['ckpt']}，將從頭開始訓練。"
            )

    # Training process
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

        recon_logits, dist_pred = model(inputs)
        loss, cons_loss, pre_loss = loss_fn_for_train(
            recon_logits, inputs, dist_pred, dist_labels, hyper_params["alpha"]
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % hyper_params["dispLossStep"] == 0:
            train_logger.info(
                f"步驟 [{step}/{hyper_params['maxIter']}], 總損失: {loss.item():.4f}, "
                f"重建損失: {cons_loss.item():.4f}, 距離場損失: {pre_loss.item():.4f}"
            )
            writer.add_scalar("Loss/train_total", loss.item(), step)
            writer.add_scalar("Loss/train_reconstruction", cons_loss.item(), step)
            writer.add_scalar("Loss/train_distance_field", pre_loss.item(), step)

        if step > 0 and step % hyper_params["exeValStep"] == 0:
            model.eval()
            total_val_loss, total_val_acc = 0.0, 0.0
            with torch.no_grad():
                for i, (val_inputs, val_dist_labels) in enumerate(valid_loader):
                    val_inputs = val_inputs.to(device)

                    val_recon_logits, _ = model(val_inputs)
                    val_loss, val_acc = loss_fn_for_eval(val_recon_logits, val_inputs)
                    total_val_loss += val_loss.item()
                    total_val_acc += val_acc

                    if i == 0:
                        val_recon_imgs = torch.sigmoid(val_recon_logits)
                        grid = make_grid(
                            torch.cat([val_inputs[:8].cpu(), val_recon_imgs[:8].cpu()]),
                            nrow=8,
                        )
                        writer.add_image("Validation/reconstruction", grid, step)

            avg_val_loss = total_val_loss / len(valid_loader)
            avg_val_acc = total_val_acc / len(valid_loader)
            writer.add_scalar("Loss/validation", avg_val_loss, step)
            writer.add_scalar("Accuracy/validation", avg_val_acc, step)
            train_logger.info(
                f"--- 驗證步驟 {step} --- 平均損失: {avg_val_loss:.4f}, 平均準確率: {avg_val_acc:.4f}"
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


def test_net(logger, output_folder):
    """測試網路並輸出最終評估指標"""
    test_logger = logging.getLogger("main.testing")
    test_logger.info("---Begin testing: ---")
    device = torch.device(hyper_params["device"])

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
            loss, acc = loss_fn_for_eval(recon_logits, inputs)
            total_loss += loss.item()
            total_acc += acc

    avg_loss = total_loss / len(test_loader)
    avg_acc = total_acc / len(test_loader)
    test_logger.info(f"測試完成 -> 平均損失: {avg_loss:.4f}, 平均準確率: {avg_acc:.4f}")


def output_vis(logger, output_folder):
    """視覺化輸出，儲存原始圖與重建圖"""
    vis_logger = logging.getLogger("main.visualizing")
    vis_logger.info("---Begin visualizing: ---")
    device = torch.device(hyper_params["device"])

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

    img_folder = os.path.join(output_folder, "imgs")
    os.makedirs(img_folder, exist_ok=True)

    with torch.no_grad():
        for i, (inputs, _) in enumerate(vis_loader):
            if i >= 3000:
                break  # 與原始碼限制一致

            inputs = inputs.to(device)
            recon_logits, _ = model(inputs)
            recon_imgs = torch.sigmoid(recon_logits)

            # 儲存原始圖和重建圖
            save_image(inputs, os.path.join(img_folder, f"{i}_original.jpeg"))
            save_image(recon_imgs, os.path.join(img_folder, f"{i}_reconstructed.jpeg"))

            if (i + 1) % 100 == 0:
                vis_logger.info(f"已儲存 {i+1} 張圖像...")

    vis_logger.info(f"視覺化圖像已儲存至: {img_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--devices", help="GPU device indices (e.g., 0 or 0,1)", type=str, default="0"
    )
    parser.add_argument(
        "--ckpt",
        help="checkpoint path",
        type=str,
        default="Embedding_Network_Model_Weight.pth",
    )
    parser.add_argument("--cnt", help="continue training flag", action="store_true")
    parser.add_argument("--rootFt", help="root feature size", type=int, default=32)
    parser.add_argument(
        "--status",
        help="training or testing flag (train, test, vis)",
        type=str,
        default="train",
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

    # 設定輸出資料夾
    timeSufix = time.strftime(r"%Y%m%d_%H%M%S")
    output_folder = os.path.join(hyper_params["outDir"], f"_{timeSufix}")
    os.makedirs(output_folder, exist_ok=True)

    # 設定 Logger
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

    # 根據狀態開始執行
    if hyper_params["status"] == "train":
        train_net(logger, output_folder)
    elif hyper_params["status"] == "test":

        test_net(logger, output_folder)
    elif hyper_params["status"] == "vis":
        output_vis(logger, output_folder)
    else:
        logger.error(
            f"未知的狀態: {hyper_params['status']}. 請使用 'train', 'test', 或 'vis'."
        )
