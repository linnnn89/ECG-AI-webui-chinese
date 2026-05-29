# code/train_5cls_v2.py
import os, argparse, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from datasets.ptbxl import PTBXLWaveform
from models.inception_time import InceptionTime

def macro_auc(y_true, y_score):
    """计算多标签的平均 AUC"""
    try:
        return roc_auc_score(y_true, y_score, average='macro')
    except:
        return 0.5

def macro_f1(y_true, y_score, threshold=0.5):
    """计算多标签的平均 F1"""
    y_pred = (y_score > threshold).astype(float)
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    # 1. 加载数据
    train_ds = PTBXLWaveform(args.ptb, split="train", cache=True)
    val_ds   = PTBXLWaveform(args.ptb, split="val",   cache=True)
    classes  = train_ds.classes
    c_out    = len(classes)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    print(f"📦 数据加载完成 | 训练集: {len(train_ds)} | 验证集: {len(val_ds)} | 类别: {classes}")

    # 2. 模型初始化
    model = InceptionTime(c_in=12, c_out=c_out, n_blocks=6, out_ch=32, bottleneck=32, dropout=0.2).to(device)
    
    # if hasattr(torch, 'compile') and device.type == 'cuda':
    #     print("🚀 正在编译模型以加速训练...")
    #     model = torch.compile(model)

    # 3. 损失函数与优化器
    y_train = np.stack([train_ds.meta[c].values for c in classes], 1).astype(np.float32)
    pos = y_train.sum(0); neg = len(train_ds) - pos + 1e-6
    pos_weight = torch.tensor(neg / (pos+1e-6), dtype=torch.float32, device=device)
    
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = len(train_dl) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda')

    best_auc = -1.0
    best_path = os.path.join(args.out, "inception_5cls_best.pt")
    os.makedirs(args.out, exist_ok=True)

    # 4. 主训练循环
    print("🔥 开始训练...")
    for epoch in range(1, args.epochs+1):
        # --- 训练阶段 ---
        model.train()
        t0, train_loss_sum, n = time.time(), 0.0, 0
        
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=args.amp, dtype=torch.float16):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step() 
            
            train_loss_sum += loss.detach().item() * len(xb)
            n += len(xb)
            
        train_loss = train_loss_sum / n
        
        # --- 验证阶段 ---
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        yp_list, yt_list = [], []
        
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda', enabled=args.amp, dtype=torch.float16):
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    
                val_loss_sum += loss.item() * len(xb)
                val_n += len(xb)
                
                prob = torch.sigmoid(logits).cpu().numpy()
                yp_list.append(prob)
                yt_list.append(yb.cpu().numpy())
                
        val_loss = val_loss_sum / val_n
        y_true = np.concatenate(yt_list, 0)
        y_prob = np.concatenate(yp_list, 0)
        
        val_auc = macro_auc(y_true, y_prob)
        val_f1  = macro_f1(y_true, y_prob, threshold=0.5)

        print(f"Epoch [{epoch}/{args.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} | Time: {time.time()-t0:.1f}s")

        # --- 保存最优模型 ---
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model": model.state_dict(), "classes": classes}, best_path)
            print(f"  🌟 发现更优模型！已保存至 {best_path} (AUC: {best_auc:.4f})")

    # 训练结束，保存最后一轮的模型
    last_path = os.path.join(args.out, "inception_5cls_last.pt")
    torch.save({"model": model.state_dict(), "classes": classes}, last_path)
    print(f"🎉 训练全部完成！最佳 AUC: {best_auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练心电图分类模型")
    parser.add_argument('--ptb', type=str, required=True, help='PTB-XL 数据集根目录')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--amp', action='store_true', help='开启自动混合精度加速')
    parser.add_argument('--out', type=str, default='O:/ECG AI project/models', help='保存模型的文件夹')
    
    args = parser.parse_args()
    main(args)