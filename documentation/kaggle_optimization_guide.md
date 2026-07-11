# ALM v12.0 Kaggle Optimization Workflow

Since ALM v12 does not utilize neural backpropagation training (the AWM relies on frozen foundation models and a deterministic logic graph), the Kaggle workflow is dedicated to **Hyperparameter Grid Search Optimization**.

We use Kaggle's free GPUs not for training weights, but for massively parallelizing inference over the `MultimodalDatasetBuilder` scenes to discover the optimal logic bounds (e.g., temporal thresholds, belief momentum) in `reasoning_engine/config.py`.

## The Kaggle Workflow

### 1. Local Preparation
1. Ensure your local `training/dataset_builder.py` and `training/optimize.py` are functioning.
2. Zip the entire `alm-project` repository (excluding `.git`, `venv`, and huge files matching the `.gitignore` policy).
3. Upload `alm-project.zip` to a private Kaggle Dataset.

### 2. Kaggle Notebook Initialization
1. Create a new Notebook on Kaggle and select **GPU T4 x2**.
2. Attach the `alm-project` dataset to your notebook.
3. Attach the public `FSD50K` and `LibriSpeech` datasets to your notebook.

### 3. Execution Script
Run the following in a Kaggle notebook cell:

```bash
# 1. Extract Project
!unzip /kaggle/input/alm-project-dataset/alm-project.zip -d /kaggle/working/alm-project

# 2. Install Dependencies
!cd /kaggle/working/alm-project && pip install -r requirements.txt

# 3. Generate Evaluation Dataset
# (Assuming dataset_builder.py is hooked up to Kaggle input paths)
!cd /kaggle/working/alm-project && python -c "from training.dataset_builder import MultimodalDatasetBuilder; builder = MultimodalDatasetBuilder('/kaggle/working/synth_dataset'); # build scenes..."

# 4. Run Grid Search Optimization
!cd /kaggle/working/alm-project && python training/optimize.py
```

### 4. Artifact Retrieval
Once the grid search completes, it will export an `artifacts/optimal_bounds.json` file.
1. Download this JSON file from the Kaggle Output directory.
2. Locally, manually merge these bounds into your `reasoning_engine/config.py`.
3. Push the new configuration to GitHub.

## Why this Workflow?
Kaggle provides free disk space and GPU inference speed. While we aren't using the GPU for PyTorch `loss.backward()`, we ARE using it to rapidly extract Whisper and HTS-AT embeddings over 10,000+ synthesized overlapping audio scenes during the Grid Search. Doing this on a local MacBook CPU would take days; on a Kaggle T4, it takes hours.
