import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from core.fusion_layer import FusionLayer
from core.scene_network import SceneContextNetwork

class ALMMultiLabelDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        w_emb, c_emb = self.embeddings[idx]
        # Explicitly cast to float32 numpy arrays to fix object_ dtype issues
        w_emb = np.array(w_emb, dtype=np.float32)
        c_emb = np.array(c_emb, dtype=np.float32)
        label = np.array(self.labels[idx], dtype=np.float32)
        return torch.tensor(w_emb), torch.tensor(c_emb), torch.tensor(label)

def load_processed_data(data_path="data/processed"):
    """Loads pre-extracted embeddings from dataset_builder.py"""
    embeddings_path = os.path.join(data_path, "embeddings.npy")
    labels_path = os.path.join(data_path, "labels.npy")
    
    if not os.path.exists(embeddings_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}. Run dataset_builder.py first.")
        
    embeddings = np.load(embeddings_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    return embeddings, labels

def train_model(data_path: str = "data/processed",
                save_path: str = "models/scene_model.pt",
                batch_size: int = 32,
                epochs: int = 50,
                lr: float = 1e-4,
                weight_decay: float = 1e-4):
    """
    Train the v4.0 Cross-Attention Fusion layer and Scene Context Network.
    """
    print(f"Loading real multi-label embeddings from {data_path}...")
    embeddings, labels = load_processed_data(data_path)
    X_train, X_val, y_train, y_val = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    
    train_dataset = ALMMultiLabelDataset(X_train, y_train)
    val_dataset = ALMMultiLabelDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    fusion = FusionLayer().to(device)
    scene_net = SceneContextNetwork(num_classes=20).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(list(fusion.parameters()) + list(scene_net.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_f1 = 0.0
    patience = 10
    patience_counter = 0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    for epoch in range(epochs):
        fusion.train()
        scene_net.train()
        train_loss = 0.0
        
        for w_emb, c_emb, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            w_emb, c_emb, batch_labels = w_emb.to(device), c_emb.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            
            fused = fusion(w_emb, c_emb)
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
            for w_emb, c_emb, batch_labels in val_loader:
                w_emb, c_emb, batch_labels = w_emb.to(device), c_emb.to(device), batch_labels.to(device)
                fused = fusion(w_emb, c_emb)
                logits = scene_net(fused)
                loss = criterion(logits, batch_labels)
                
                val_loss += loss.item() * w_emb.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.30).float()
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch_labels.cpu().numpy())
        
        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        prec = precision_score(all_labels, all_preds, average='micro', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='micro', zero_division=0)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
        
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
            print(f"Best model saved to {save_path} (F1: {f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    print(f"Training complete! Best F1 Score: {best_f1:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/processed", help="Path to processed data")
    parser.add_argument("--save_path", type=str, default="models/scene_model.pt", help="Path to save trained v4 model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    args = parser.parse_args()
    train_model(args.data_path, args.save_path, args.batch_size, args.epochs, args.lr, args.weight_decay)
