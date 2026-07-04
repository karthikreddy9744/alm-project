import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from core_modules.fusion_layer import FusionLayer
from core_modules.scene_network import SceneContextNetwork
from training.train import ALMMultiLabelDataset, load_processed_data
from core_modules.scene_network import SCENE_LABELS

def evaluate_model(data_path: str = "data/processed",
                   model_path: str = "models/alm_v10_final.pt",
                   output_path: str = "data/evaluation"):
    """
    Evaluate trained multi-label model on real embeddings and generate metrics.
    """
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Loading real multi-label embeddings from {data_path} for evaluation...")
    embeddings, labels = load_processed_data(data_path)
    
    # Create dataset and dataloader
    dataset = ALMMultiLabelDataset(embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Load trained models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion = FusionLayer().to(device)
    scene_net = SceneContextNetwork(num_classes=40).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    fusion.load_state_dict(checkpoint["fusion"])
    scene_net.load_state_dict(checkpoint["scene_net"])
    
    fusion.eval()
    scene_net.eval()
    
    # Run inference
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for w_emb, c_emb, batch_labels in dataloader:
            w_emb, c_emb, batch_labels = w_emb.to(device), c_emb.to(device), batch_labels.to(device)
            fused = fusion(w_emb, c_emb)
            logits = scene_net(fused)
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.30).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())
            
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate metrics
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    prec = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=SCENE_LABELS, zero_division=0)
    
    # Print metrics
    print("=== Evaluation Results ===")
    print(f"Micro F1: {f1_micro:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Precision (Micro): {prec:.4f}")
    print(f"Recall (Micro): {rec:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Save metrics
    with open(os.path.join(output_path, "metrics.txt"), "w") as f:
        f.write(f"Micro F1: {f1_micro:.4f}\n")
        f.write(f"Macro F1: {f1_macro:.4f}\n")
        f.write(f"Precision (Micro): {prec:.4f}\n")
        f.write(f"Recall (Micro): {rec:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"Metrics saved to {os.path.join(output_path, 'metrics.txt')}")
    
    return f1_micro, f1_macro, report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/processed", help="Path to processed data")
    parser.add_argument("--model_path", type=str, default="models/scene_model.pt", help="Path to trained model")
    parser.add_argument("--output_path", type=str, default="data/evaluation", help="Path to save evaluation results")
    args = parser.parse_args()
    evaluate_model(args.data_path, args.model_path, args.output_path)
