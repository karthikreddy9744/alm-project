import torch.nn as nn

SCENE_LABELS = [
    "Dog", "Poultry", "Pig", "Cow", "Frog",
    "Cat", "Insects", "Sheep", "Crow", "Rain & Thunder",
    "Sea & Water", "Crackling fire", "Crickets", "Chirping birds", "Water drops",
    "Wind", "Toilet flush", "Crying baby", "Coughing & Sneezing", "Clapping",
    "Breathing", "Footsteps", "Laughing", "Personal Care", "Snoring",
    "Door sounds", "Office", "Can opening", "Washing machine", "Vacuum cleaner",
    "Clock", "Glass breaking", "Aviation", "Saws", "Siren",
    "Cars/Traffic", "Train", "Church bells", "Fireworks", "Silence / Unknown"
]

class SceneContextNetwork(nn.Module):
    """
    ALM v7.0 Scene Context Network.
    Simplified 2-layer MLP to map from Joint Representation (256) to 40 scene categories.
    Architecture: [256 -> 128 -> 40 classes]
    """
    def __init__(self, input_dim=256, num_classes=40):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(x) # logits [B, num_classes]