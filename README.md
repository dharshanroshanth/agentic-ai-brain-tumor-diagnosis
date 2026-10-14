# Brain Tumor Diagnosis Assistant

An Agentic AI system for brain tumor diagnosis from MRI scans.

## Project Structure

```
mini project/
├── requirements.txt          # All dependencies
├── pipeline.py               # Main diagnostic pipeline (run this)
├── train_kfold.py            # 5-fold cross-validation training
├── evaluate_external.py      # External validation across all 3 test sets
├── checkpoints/              # Model weights saved per fold
├── results/                  # Confusion matrices, metric reports
├── data/
│   ├── config.py             # All dataset path mappings
│   └── dataset.py            # PyTorch Dataset classes + augmentation
├── agents/
│   ├── perception.py         # EfficientNetV2-S + Uncertainty Gate
│   ├── vision_explainability.py  # U-Net segmentation + Grad-CAM
│   ├── orchestrator.py       # LangChain LLM Orchestrator
│   ├── verifier.py           # RAG-based diagnosis verifier
│   └── report_generator.py   # Clinical report synthesizer
└── notebooks/
    └── colab_training.py     # Colab training setup guide
```

## Datasets Used (auto-detected from `dataset/` folder)

| Sub-Dataset               | Role                                                   |
| ------------------------- | ------------------------------------------------------ |
| `Training/` (or `Train/`) | Training set                                           |
| `Testing/` (or `Test/`)   | In-distribution test set                               |
| `Val/`                    | Validation set (optional, used in some configurations) |

Inside each of these folders, the model expects to find four sub-directories representing the target classes: `glioma`, `meningioma`, `pituitary`, and `no_tumor`.

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Perception Agent (K-Fold)

```bash
python train_kfold.py
```

### 3. Run External Validation

```bash
python evaluate_external.py
```

### 4. Run the Full Diagnostic Pipeline

```bash
python pipeline.py
```

### 5. On Google Colab (T4 GPU)

Open `notebooks/colab_training.py` and copy each cell section into Colab code cells.

### 6. Run the Streamlit App

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

## LLM Configuration

Set your OpenAI API key before running the pipeline:

```bash
set OPENAI_API_KEY=your_key_here   # Windows
export OPENAI_API_KEY=your_key     # Linux/Colab
```

## Evaluation Metrics Reported

- Accuracy, Precision, Recall, F1 (per-class and weighted avg)
- Mean ± Std across 5 folds
- Confusion matrices per fold
- Model parameters, FLOPs, inference latency
- Safety Gate escalation rate
