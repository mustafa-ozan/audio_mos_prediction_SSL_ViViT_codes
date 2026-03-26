import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import random
import gc
import winsound
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from scipy.stats import spearmanr

# --- Stability Config ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# --- 1. Paths & Global Config ---
BASE_DIR = "D:/BUU_PC_DATA/audio_mos_prediction_projectt/"
MFCC_DIR = os.path.join(BASE_DIR, "data/MFCC_data/")
EXP_ROOTS = [
    os.path.join(BASE_DIR, "experiments"),
    os.path.join(BASE_DIR, "experiments_only_english")
]

BATCH_SIZE = 64
ACCUMULATION_STEPS = 2  # Effective batch size = 16
LEARNING_RATE = 1e-5
NUM_EPOCHS = 500
PATIENCE = 50


# --- 2. Model Architecture ---
class CDS_ViViT(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=(10, 31), stride=(10, 31))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.mos_head = nn.Linear(embed_dim, 1)

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if mask is not None:
            cls_mask = torch.zeros((mask.shape[0], 1), device=mask.device, dtype=torch.bool)
            full_mask = torch.cat([cls_mask, mask], dim=1)
            x = self.transformer(x, src_key_padding_mask=full_mask)
        else:
            x = self.transformer(x)
        return self.mos_head(x[:, 0])


# --- 3. Dataset & Collate ---
class MFCCDataset(Dataset):
    def __init__(self, df, base_dir):
        self.df = df
        self.base_dir = base_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        wav_name = str(self.df.iloc[idx, 0])
        audio_id = wav_name.replace('.wav', '')
        mos_label = float(self.df.iloc[idx, 1])
        folder_path = os.path.join(self.base_dir, audio_id)
        try:
            meta_df = pd.read_csv(os.path.join(folder_path, "segment_info.csv")).sort_values('segment_path')
            mfccs, masks = [], []
            for _, row in meta_df.iterrows():
                mfcc = np.load(os.path.join(folder_path, row['segment_path']))
                mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
                mfccs.append(torch.from_numpy(mfcc).float())
                valid_slices = int(np.ceil(int(row['valid_frames']) / 31))
                m = torch.zeros((4, 10), dtype=torch.bool)
                m[:, valid_slices:] = True
                masks.append(m.flatten())
            return torch.cat(mfccs, dim=1), torch.tensor([mos_label]).float(), torch.cat(masks)
        except:
            return self.__getitem__(random.randint(0, len(self.df) - 1))


def collate_cds(batch):
    mfccs, labels, masks = zip(*batch)
    padded_mfccs = pad_sequence([m.T for m in mfccs], batch_first=True).transpose(1, 2)
    padded_masks = pad_sequence(masks, batch_first=True, padding_value=True)
    return padded_mfccs, torch.stack(labels), padded_masks


def play_notification_sound(sound_type="epoch"):
    try:
        if sound_type == "epoch":
            winsound.Beep(1000, 200);
            winsound.Beep(1200, 200)
        elif sound_type == "success":
            [winsound.Beep(f, 300) for f in [440, 554, 659, 880]]
    except:
        pass


def get_latest_checkpoint(ckpt_path, emergency_path):
    ckpt = torch.load(ckpt_path, map_location='cuda') if os.path.exists(ckpt_path) else None
    emergency = torch.load(emergency_path, map_location='cuda') if os.path.exists(emergency_path) else None
    if ckpt and emergency:
        if emergency.get('epoch', 0) > ckpt.get('epoch', 0): return emergency
        return emergency if emergency.get('batch_idx', 0) > ckpt.get('batch_idx', 0) else ckpt
    return ckpt if ckpt else emergency


# --- 4. Training Engine ---
def run_experiment(exp_path, is_english_only=False):
    exp_name = os.path.basename(exp_path)
    suffix = "" if is_english_only else "_split"

    train_file = os.path.join(exp_path, f"train{suffix}.csv")
    val_file = os.path.join(exp_path, f"val{suffix}.csv")
    test_file = os.path.join(exp_path, f"test{suffix}.csv")
    ckpt_dir = os.path.join(exp_path, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Checkpoint filenames
    best_path = os.path.join(ckpt_dir, "best_model_VIVIT.pth")
    resume_path = os.path.join(ckpt_dir, "resume_checkpoint_VIVIT.pth")
    emergency_path = os.path.join(ckpt_dir, "emergency_VIVIT.pth")
    log_path = os.path.join(exp_path, "training_log_VIVIT.csv")

    device = torch.device("cuda")
    model = CDS_ViViT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss();
    scaler = GradScaler('cuda')

    # Load Logic
    latest_ckpt = get_latest_checkpoint(resume_path, emergency_path)
    start_epoch, start_batch, best_val_loss, history, wait = 0, 0, float('inf'), [], 0

    if latest_ckpt:
        model.load_state_dict(latest_ckpt['model_state_dict'])
        optimizer.load_state_dict(latest_ckpt['optimizer_state_dict'])
        start_epoch = latest_ckpt['epoch']
        start_batch = latest_ckpt.get('batch_idx', 0) + 1
        history = latest_ckpt.get('history', [])
        if history:
            val_losses = [h['val_loss'] for h in history]
            best_val_loss = min(val_losses)
            wait = 0
            for v in reversed(val_losses):
                if v > best_val_loss:
                    wait += 1
                else:
                    break
        print(f"🔄 Resuming VIVIT {exp_name} | E{start_epoch} | Wait: {wait}/{PATIENCE}")

    if wait >= PATIENCE: #or os.path.exists(os.path.join(exp_path, "test_predictions_VIVIT.csv")):
        print(f"⏩ Skipping {exp_name}: Completed or Early Stopped.")
        return

    train_loader = DataLoader(MFCCDataset(pd.read_csv(train_file), MFCC_DIR), batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_cds, num_workers=2, pin_memory=True)
    val_loader = DataLoader(MFCCDataset(pd.read_csv(val_file), MFCC_DIR), batch_size=BATCH_SIZE, collate_fn=collate_cds,
                            num_workers=2, pin_memory=True)

    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            gc.collect();
            torch.cuda.empty_cache();
            torch.cuda.synchronize()
            model.train();
            running_loss = 0.0
            loop = tqdm(train_loader, desc=f"{exp_name} | E{epoch}")

            optimizer.zero_grad(set_to_none=True)
            for i, (inputs, labels, masks) in enumerate(loop):
                if epoch == start_epoch and i < start_batch: continue
                inputs, labels, masks = inputs.to(device, non_blocking=True), labels.to(device,
                                                                                        non_blocking=True), masks.to(
                    device, non_blocking=True)

                with autocast('cuda'):
                    loss = criterion(model(inputs, mask=masks), labels) / ACCUMULATION_STEPS

                scaler.scale(loss).backward()

                if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                    scaler.step(optimizer);
                    scaler.update();
                    optimizer.zero_grad(set_to_none=True)
                    if (i + 1) % 100 == 0: torch.cuda.empty_cache()

                running_loss += (loss.detach().item() * ACCUMULATION_STEPS)
                if i % 1000 == 0:
                    torch.save({'epoch': epoch, 'batch_idx': i, 'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(), 'history': history}, emergency_path)
                del inputs, labels, masks, loss

            # Validation
            start_batch = 0;
            model.eval();
            val_loss = 0.0
            with torch.inference_mode():
                for i, (iv, lv, mv) in enumerate(tqdm(val_loader, desc="Val")):
                    with autocast('cuda'):
                        v_preds = model(iv.to(device), mask=mv.to(device))
                        v_loss = criterion(v_preds, lv.to(device))
                        val_loss += v_loss.item()
                    del iv, lv, mv, v_preds, v_loss
                    if (i + 1) % 50 == 0: torch.cuda.empty_cache()

            avg_train = running_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            print(f"✅ E{epoch} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | Best: {best_val_loss:.4f}")
            play_notification_sound("epoch")

            if avg_val < best_val_loss:
                best_val_loss, wait = avg_val, 0
                torch.save(model.state_dict(), best_path)
            else:
                wait += 1

            history.append({'epoch': epoch, 'train_loss': avg_train, 'val_loss': avg_val, 'wait': wait})
            pd.DataFrame(history).to_csv(log_path, index=False)
            torch.save({'epoch': epoch + 1, 'batch_idx': 0, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'history': history}, resume_path)

            if wait >= PATIENCE: break

    except Exception as e:
        print(f"❌ Error in {exp_name}: {e}")


if __name__ == "__main__":
    for root in EXP_ROOTS:
        is_eng = "english" in root
        subdirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        for s in sorted(subdirs): run_experiment(s, is_english_only=is_eng)