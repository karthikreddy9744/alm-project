import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from core.fusion_layer import FusionLayer
from core.scene_network import SceneContextNetwork
from training.train import ALMDataset, load_processed_data
from core.context_builder import SCENE_LABELS

def evaluate_model(data_path: str = "data/processed",
                   model_path: str = "models/scene_model.pt",
                   output_path: str = "data/evaluation"):
    """
    Evaluate trained model and generate metrics.
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Load data
    embeddings, labels = load_processed_data(data_path)
    
    # Create dataset and dataloader
    dataset = ALMDataset(embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Load trained models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion = FusionLayer().to(device)
    scene_net = SceneContextNetwork().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    fusion.load_state_dict(checkpoint["fusion"])
    scene_net.load_state_dict(checkpoint["scene_net"])
    
    fusion.eval()
    scene_net.eval()
    
    # Run inference
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for w_emb, c_emb, labels in dataloader:
            w_emb, c_emb, labels = w_emb.to(device), c_emb.to(device), labels.to(device)
            fused = fusion(w_emb, c_emb)
            logits = scene_net(fused)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', labels=range(len(SCENE_LABELS)), zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=SCENE_LABELS, labels=range(len(SCENE_LABELS)), zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(SCENE_LABELS)))
    
    # Print metrics
    print("=== Evaluation Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Save metrics
    with open(os.path.join(output_path, "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1: {f1_macro:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=SCENE_LABELS, yticklabels=SCENE_LABELS)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "confusion_matrix.png"))
    print(f"Confusion matrix saved to {os.path.join(output_path, 'confusion_matrix.png')}")
    
    return accuracy, f1_macro, report, cm

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/processed", help="Path to processed data")
    parser.add_argument("--model_path", type=str, default="models/scene_model.pt", help="Path to trained model")
    parser.add_argument("--output_path", type=str, default="data/evaluation", help="Path to save evaluation results")
    args = parser.parse_args()
    evaluate_model(args.data_path, args.model_path, args.output_path)
