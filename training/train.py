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
        # multi-hot encoding
        return torch.tensor(w_emb, dtype=torch.float32), torch.tensor(c_emb, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

def generate_simulated_multilabel_data(num_samples=2000, num_classes=20):
    """
    Simulates multi-label dataset for v4.0 architecture training.
    Real deployment would mix AudioSet, LibriSpeech, and ESC-50 here.
    """
    embeddings = []
    labels = []
    for _ in range(num_samples):
        # random Whisper/CLAP embeddings
        w = np.random.randn(512)
        c = np.random.randn(512)
        embeddings.append((w, c))
        
        # 1-3 active labels per audio clip
        multi_hot = np.zeros(num_classes)
        active_indices = np.random.choice(num_classes, size=np.random.randint(1, 4), replace=False)
        multi_hot[active_indices] = 1.0
        labels.append(multi_hot)
        
    return embeddings, np.array(labels)

def train_model(save_path: str = "models/scene_model.pt",
                batch_size: int = 32,
                epochs: int = 50,
                lr: float = 1e-4, # lower LR for attention
                weight_decay: float = 1e-4):
    """
    Train the v4.0 Cross-Attention Fusion layer and Scene Context Network using BCEWithLogitsLoss.
    """
    # 1. Load Data (Simulated for architecture upgrade proof of concept)
    print("Generating simulated multi-label dataset mixtures...")
    embeddings, labels = generate_simulated_multilabel_data(num_samples=5000, num_classes=20)
    X_train, X_val, y_train, y_val = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    
    # Create datasets and dataloaders
    train_dataset = ALMMultiLabelDataset(X_train, y_train)
    val_dataset = ALMMultiLabelDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    fusion = FusionLayer().to(device)
    scene_net = SceneContextNetwork(num_classes=20).to(device)
    
    # Multi-label loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(list(fusion.parameters()) + list(scene_net.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
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
        
        # Validation
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
        
        # Calculate metrics
        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        f1 = f1_score(all_labels, all_preds, average='micro')
        prec = precision_score(all_labels, all_preds, average='micro', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='micro', zero_division=0)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
        
        # Save best model
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
    parser.add_argument("--save_path", type=str, default="models/scene_model.pt", help="Path to save trained v4 model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    args = parser.parse_args()
    train_model(args.save_path, args.batch_size, args.epochs, args.lr, args.weight_decay)
