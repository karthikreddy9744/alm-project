import os
import json
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from core_modules.fusion_layer import FusionLayer
from core_modules.scene_network import SceneContextNetwork

class UnifiedALMDataset(Dataset):
    def __init__(self, embeddings, labels_with_stage):
        self.embeddings = embeddings
        self.labels_with_stage = labels_with_stage
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        w_emb, c_emb, h_emb = self.embeddings[idx]
        w_emb = np.array(w_emb, dtype=np.float32)
        c_emb = np.array(c_emb, dtype=np.float32)
        h_emb = np.array(h_emb, dtype=np.float32)
        
        # The first 40 dims are the multi-hot labels, the last (41st) is the curriculum stage
        label_full = np.array(self.labels_with_stage[idx], dtype=np.float32)
        multi_hot = label_full[:40]
        stage = int(label_full[40])
        
        return torch.tensor(w_emb), torch.tensor(c_emb), torch.tensor(h_emb), torch.tensor(multi_hot), stage

def load_processed_data(data_path="data/processed"):
    emb_files = sorted(glob.glob(os.path.join(data_path, "embeddings_shard_*.npy")))
    lbl_files = sorted(glob.glob(os.path.join(data_path, "labels_shard_*.npy")))
    
    # Fallback to older non-sharded files if they exist
    if not emb_files:
        emb_files = [os.path.join(data_path, "embeddings.npy")]
        lbl_files = [os.path.join(data_path, "labels.npy")]
        
    if not os.path.exists(emb_files[0]):
        raise FileNotFoundError(f"No processed data found at {data_path}. Run dataset_builder.py first.")
        
    all_embeddings = []
    all_labels = []
    
    print(f"Loading {len(emb_files)} data shards...")
    for emb_f, lbl_f in zip(emb_files, lbl_files):
        all_embeddings.extend(np.load(emb_f, allow_pickle=True))
        all_labels.extend(np.load(lbl_f, allow_pickle=True))
        
    # If the user ran an older dataset builder without stages, append stage 4 (complex) to everything
    if len(all_labels[0]) == 40:
        print("Warning: Older dataset format detected (No curriculum stages). Assuming all samples are Stage 4.")
        all_labels = [np.append(lbl, [4]) for lbl in all_labels]
        
    return all_embeddings, all_labels

def get_curriculum_indices(dataset: UnifiedALMDataset, allowed_stages: list) -> list:
    indices = []
    for idx in range(len(dataset)):
        stage = dataset.labels_with_stage[idx][40]
        if stage in allowed_stages:
            indices.append(idx)
    return indices

def train_model(data_path: str = "data/processed",
                save_path: str = "models/scene_model.pt",
                batch_size: int = 32,
                epochs: int = 50,
                lr: float = 1e-4,
                weight_decay: float = 1e-4):
    """
    Train the ALM v12.7.1 Unified Architecture with Curriculum Learning.
    """
    embeddings, labels = load_processed_data(data_path)
    X_train, X_val, y_train, y_val = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    
    train_dataset = UnifiedALMDataset(X_train, y_train)
    val_dataset = UnifiedALMDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    fusion = FusionLayer().to(device)
    scene_net = SceneContextNetwork(num_classes=40).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(list(fusion.parameters()) + list(scene_net.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_f1 = 0.0
    patience = 10
    patience_counter = 0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    for epoch in range(epochs):
        # ---------------------------------------------------------
        # ALM v12.7.1 CURRICULUM LEARNING LOGIC
        # ---------------------------------------------------------
        if epoch < 10:
            # Stage 1: Clean Speech (6), Environment Only (5), 1 Spk + 1 Env (1)
            allowed_stages = [1, 5, 6]
            curr_name = "Curriculum Phase 1 (Clean/Simple)"
        elif epoch < 20:
            # Stage 2: Unlock 1 Spk + 2-3 Env (2)
            allowed_stages = [1, 2, 5, 6]
            curr_name = "Curriculum Phase 2 (Moderate Polyphony)"
        elif epoch < 30:
            # Stage 3: Unlock 2 Spk + 2-3 Env (3)
            allowed_stages = [1, 2, 3, 5, 6]
            curr_name = "Curriculum Phase 3 (Multi-Speaker)"
        else:
            # Stage 4: Unlock 3 Spk + 3-5 Env (4). Full Dataset.
            allowed_stages = [1, 2, 3, 4, 5, 6]
            curr_name = "Curriculum Phase 4 (Full Complexity)"
            
        curriculum_indices = get_curriculum_indices(train_dataset, allowed_stages)
        active_train_dataset = Subset(train_dataset, curriculum_indices)
        train_loader = DataLoader(active_train_dataset, batch_size=batch_size, shuffle=True)
        
        print(f"\n--- Epoch {epoch+1}/{epochs} | {curr_name} | Active Samples: {len(active_train_dataset)} ---")
        
        fusion.train()
        scene_net.train()
        train_loss = 0.0
        
        for w_emb, c_emb, h_emb, batch_labels, _ in tqdm(train_loader, desc="Training"):
            w_emb, c_emb, h_emb = w_emb.to(device), c_emb.to(device), h_emb.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            fused = fusion(w_emb, c_emb, h_emb)
            logits = scene_net(fused)
            loss = criterion(logits, batch_labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * w_emb.size(0)
            
        scheduler.step()
        
        fusion.eval()
        scene_net.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for w_emb, c_emb, h_emb, batch_labels, _ in val_loader:
                w_emb, c_emb, h_emb = w_emb.to(device), c_emb.to(device), h_emb.to(device)
                batch_labels = batch_labels.to(device)
                
                fused = fusion(w_emb, c_emb, h_emb)
                logits = scene_net(fused)
                loss = criterion(logits, batch_labels)
                
                val_loss += loss.item() * w_emb.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.30).float()
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch_labels.cpu().numpy())
        
        train_loss /= len(active_train_dataset)
        val_loss /= len(val_dataset)
        
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        prec = precision_score(all_labels, all_preds, average='micro', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='micro', zero_division=0)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {f1:.4f} | Val Prec: {prec:.4f} | Val Rec: {rec:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            checkpoint = {
                "fusion": fusion.state_dict(),
                "scene_net": scene_net.state_dict(),
                "epoch": epoch,
                "f1": f1
            }
            torch.save(checkpoint, save_path)
            
            metrics = {"best_f1": f1, "precision": prec, "recall": rec, "val_loss": val_loss}
            import os
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                metrics_path = os.path.join(save_dir, "training_metrics.json")
            else:
                metrics_path = "training_metrics.json"
                
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
                
            print(f"[*] Best model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    print(f"Training complete! Best F1 Score: {best_f1:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/processed", help="Path to processed data shards")
    parser.add_argument("--save_path", type=str, default="models/scene_model.pt", help="Path to save trained v12.7.1 model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    args = parser.parse_args()
    train_model(args.data_path, args.save_path, args.batch_size, args.epochs, args.lr, args.weight_decay)
