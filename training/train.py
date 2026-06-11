import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from core.fusion_layer import FusionLayer
from core.scene_network import SceneContextNetwork

class ALMDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        w_emb, c_emb = self.embeddings[idx]
        return torch.tensor(w_emb, dtype=torch.float32), torch.tensor(c_emb, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def load_processed_data(data_path: str = "data/processed"):
    embeddings = np.load(os.path.join(data_path, "embeddings.npy"), allow_pickle=True)
    labels = np.load(os.path.join(data_path, "labels.npy"), allow_pickle=True)
    return embeddings, labels

def train_model(data_path: str = "data/processed", 
                save_path: str = "models/scene_model.pt",
                batch_size: int = 32,
                epochs: int = 50,
                lr: float = 1e-3,
                weight_decay: float = 1e-4):
    """
    Train the fusion layer and scene context network.
    """
    # Load data
    embeddings, labels = load_processed_data(data_path)
    X_train, X_val, y_train, y_val = train_test_split(embeddings, labels, test_size=0.2, random_state=42, stratify=labels)
    
    # Create datasets and dataloaders
    train_dataset = ALMDataset(X_train, y_train)
    val_dataset = ALMDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion = FusionLayer().to(device)
    scene_net = SceneContextNetwork().to(device)
    
    # Loss and optimizer with class weighting
    unique_labels, counts = np.unique(labels, return_counts=True)
    # Give all 15 classes a default count of 1 to prevent division by zero for missing classes
    full_counts = np.ones(15) 
    for l, c in zip(unique_labels, counts):
        if l < 15:
            full_counts[l] = c
            
    class_weights = 1.0 / torch.tensor(full_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * 15.0
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(list(fusion.parameters()) + list(scene_net.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_val_acc = 0.0
    patience = 7
    patience_counter = 0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    for epoch in range(epochs):
        fusion.train()
        scene_net.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for w_emb, c_emb, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            w_emb, c_emb, labels = w_emb.to(device), c_emb.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            fused = fusion(w_emb, c_emb)
            logits = scene_net(fused)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * w_emb.size(0)
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        scheduler.step()
        
        # Validation
        fusion.eval()
        scene_net.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for w_emb, c_emb, labels in val_loader:
                w_emb, c_emb, labels = w_emb.to(device), c_emb.to(device), labels.to(device)
                fused = fusion(w_emb, c_emb)
                logits = scene_net(fused)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * w_emb.size(0)
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        train_loss /= train_total
        val_loss /= val_total
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint = {
                "fusion": fusion.state_dict(),
                "scene_net": scene_net.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc
            }
            torch.save(checkpoint, save_path)
            print(f"Best model saved to {save_path} (val acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/processed", help="Path to processed data")
    parser.add_argument("--save_path", type=str, default="models/scene_model.pt", help="Path to save trained model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    args = parser.parse_args()
    train_model(args.data_path, args.save_path, args.batch_size, args.epochs, args.lr, args.weight_decay)
