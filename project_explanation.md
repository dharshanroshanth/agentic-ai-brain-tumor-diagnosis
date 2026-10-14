# Brain Tumor Diagnosis Assistant: Project Overview

## 1. Project Goal & Motivation
This project is an **Agentic AI System** designed to assist radiologists in diagnosing brain tumors from MRI scans. It classifies MRI scans into four categories: **Glioma, Meningioma, Pituitary, or No Tumor**.

Instead of just being a standard image classifier, this project is built as an "Agentic Workflow." It focuses on **accuracy, explainability, and safety**, ensuring that the AI acts as an assistant that knows its own limits and can explain its reasoning, rather than a black-box decision-maker.

---

## 2. Core Architecture & Pipeline

The system is composed of several intelligent "agents" that work together in a pipeline:

### A. The Perception Agent (The "Eyes")
* **Deep Learning Ensemble:** It uses a powerful ensemble of three state-of-the-art neural networks to analyze the image:
  * **EfficientNetV2-S** (60% weight) - highly efficient CNN.
  * **DenseNet201** (20% weight) - dense feature extraction.
  * **Vision Transformer (ViT)** (20% weight) - captures global context.
* **Test-Time Augmentation (TTA):** During inference, it applies transformations (flips, rotations) to the image and averages the predictions. This maximizes accuracy and robustness.

### B. The Safety Gate (Uncertainty Estimation)
* Before making a final decision, the system calculates the **predictive entropy** (uncertainty) of its own prediction. 
* If the entropy is too high (meaning the AI is confused or uncertain), it triggers an **`ESCALATE_TO_HUMAN`** flag. This is a crucial safety feature for medical applications.

### C. Vision Explainability Agent (The "Why")
* If the AI is uncertain (escalated) or its confidence is below 90%, it automatically triggers explainability modules.
* It uses **Grad-CAM** (Gradient-weighted Class Activation Mapping) on the EfficientNet backbone and **U-Net** segmentation to generate heatmaps. This shows the doctor exactly *where* the AI is looking in the MRI scan.

### D. The Verifier & Orchestrator Agents (The "Brain")
* **Orchestrator (LangChain):** Manages the workflow, tracks the state of the diagnosis, and can simulate a chat with the radiologist.
* **Verifier Agent:** Cross-checks the AI's prediction against a database of known visual indicators (e.g., verifying if the visual features match typical textbook examples of a Glioma).

### E. Report Generator
* Finally, it synthesizes all this data into a structured **Clinical Report** detailing the findings, confidence scores, verification reasoning, and whether human review is required.

---

## 3. How Users Interact With It
The project includes a **Streamlit Web Application** (`app.py`). A radiologist can simply open the web app, upload a patient's MRI scan, and click "Run Diagnostic Pipeline." 

Within seconds, the app processes the image through the ensemble, checks the safety gate, generates explainability heatmaps if needed, and prints out a clear, structured clinical report.

---

## 4. Key Talking Points for Your Mentor
When presenting this, emphasize these advanced concepts:
1. **Not just a CNN:** You used an **Ensemble approach** combining CNNs (EfficientNet/DenseNet) with Transformers (ViT) for superior accuracy.
2. **Medical Safety First:** You implemented a **Safety Gate** based on entropy. The AI knows when it doesn't know, which is critical in healthcare.
3. **Explainable AI (XAI):** You didn't leave it as a black box; you integrated Grad-CAM so the AI's reasoning is transparent.
4. **Agentic Workflow:** You broke the problem down into discrete AI agents (Perception, Verifier, Reporter) that talk to each other, making the system highly modular and scalable.
