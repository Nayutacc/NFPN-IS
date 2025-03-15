from model.ctNet.DataLoader import MyDataset
from model.ctNet.ctNet import Nest_U_Net
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import os
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from model.mLoss import CustomLoss

# Function to compute AUC for a batch of images and masks
def compute_batch_auc(_predictions, _masks):
    # Assuming the model's output is in the form (batch_size, height, width)
    _predictions = _predictions.flatten()
    _masks = _masks.flatten()
    return roc_auc_score(_masks, _predictions)


train_txt_path = "train.txt"  # path to your train.txt file
test_txt_path = "test.txt"
image_base_path = ""  # Replace with the actual image directory path
mask_base_path = ""  # Replace with the actual mask directory path
itmask_base_path = ""

mydataset_train = MyDataset(train_txt_path, image_base_path, mask_base_path, itmask_base_path)  # 训练集数据
mydataset_test = MyDataset(test_txt_path, image_base_path, mask_base_path, itmask_base_path)  # 测试集数据
data_train = DataLoader(mydataset_train, batch_size=32, shuffle=True)
data_test = DataLoader(mydataset_test, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





epoch = 150  # 设置迭代次数

if __name__ == '__main__':
    # 初始化模型、损失函数和优化器
    mymodule = Nest_U_Net().to(device)
    loss_fn = CustomLoss(W_MSE=0.7, W_LSP=0.3)
    optimizer = optim.Adam(mymodule.parameters(), lr=0.001)

    # 创建保存路径
    model_save_dir = f"./save_model/"
    log_dir = f"./logs/"
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 日志文件路径
    loss_log_path = os.path.join(log_dir, "loss.txt")
    auc_log_path = os.path.join(log_dir, "auc.txt")
    metrics_log_path = os.path.join(log_dir, "metrics.txt")  # 新增日志文件路径

    best_miou = -1  # 初始化最好的 mIoU
    best_epoch = -1
    best_model_state = None  # 用于保存最佳模型的权重

    for i in range(epoch):
        epoch_loss = 0.0
        batch_count = 0

        # 训练阶段
        mymodule.train()
        for index, (img, mask, itmasks) in enumerate(data_train):
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            out = mymodule(img)
            loss = loss_fn(out, mask, itmasks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1

        # 保存当前 epoch 的平均损失
        avg_epoch_loss = epoch_loss / batch_count
        with open(loss_log_path, "a+", encoding="utf-8") as f:
                f.write(f"Epoch {i} - Average Loss: {avg_epoch_loss:.7f}\n")

        # 评估阶段
        mymodule.eval()
        all_aucs = []
        all_precisions = []
        all_recalls = []
        all_f1_scores = []
        all_mious = []

        with torch.no_grad():
            for batch_idx, (images, masks, itmasks) in enumerate(data_test):
                images, masks = images.to(device), masks.to(device)
                predictions = mymodule(images)
                # 将预测结果和标签转换为二进制形式（假设二分类任务）
                pred_binary = (predictions > 0.5).cpu().numpy()
                true_binary = masks.cpu().numpy()

                # 计算AUC
                batch_auc = compute_batch_auc(pred_binary, true_binary)
                all_aucs.append(batch_auc)

                # 计算精确率、召回率和F1分数
                batch_precision = precision_score(true_binary.flatten(), pred_binary.flatten())
                batch_recall = recall_score(true_binary.flatten(), pred_binary.flatten())
                batch_f1 = f1_score(true_binary.flatten(), pred_binary.flatten())
                all_precisions.append(batch_precision)
                all_recalls.append(batch_recall)
                all_f1_scores.append(batch_f1)

                # 计算mIoU
                intersection = np.sum(pred_binary * true_binary)
                union = np.sum(pred_binary) + np.sum(true_binary) - intersection
                batch_miou = intersection / union if union != 0 else 0
                all_mious.append(batch_miou)

        # 记录平均 AUC、精确率、召回率、F1 和 mIoU
        mean_auc = np.mean(all_aucs)
        mean_precision = np.mean(all_precisions)
        mean_recall = np.mean(all_recalls)
        mean_f1 = np.mean(all_f1_scores)
        mean_miou = np.mean(all_mious)

        # 写入日志文件
        with open(auc_log_path, "a+", encoding="utf-8") as f:
            f.write(f"Epoch {i} - Average AUC: {mean_auc:.7f}\n")
        with open(metrics_log_path, "a+", encoding="utf-8") as f:
            f.write(f"Epoch {i} - Precision: {mean_precision:.7f}, Recall: {mean_recall:.7f}, F1: {mean_f1:.7f}, mIoU: {mean_miou:.7f}\n")
            
        print(f"Epoch {i} - Average AUC: {mean_auc:.7f}")
        print(f"Epoch {i} - Precision: {mean_precision:.7f}, Recall: {mean_recall:.7f}, F1: {mean_f1:.7f}, mIoU: {mean_miou:.7f}")

        # 如果当前的 mIoU 比最好的 mIoU 好，更新保存的模型
        if mean_miou > best_miou:
            best_miou = mean_miou
            best_epoch = i
            best_model_state = mymodule.state_dict()  # 保存当前模型的权重

    # 保存效果最好的模型
    if best_model_state is not None:
        best_model_save_path = os.path.join(model_save_dir, f"best_model_epoch_{best_epoch}_miou_{best_miou:.7f}.pth")
        torch.save(best_model_state, best_model_save_path)
        print(f"模型训练完成，最好的模型已保存到 {best_model_save_path}")
