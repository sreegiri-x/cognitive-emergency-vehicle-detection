# Cognitive EV Detection

**Cognitive Emergency Vehicle Detection and Risk Analysis using Explainable AI**

An AI-based computer vision framework for detecting and analyzing emergency-vehicle-related traffic situations from real-world highway video data. The project combines video understanding, deep learning, and explainable AI to identify potentially critical traffic events and provide visual explanations of model decisions.

> **Project Status:** Active Development

---

## Overview

Emergency vehicles operating in highway traffic can create complex and potentially hazardous interactions with surrounding vehicles. Detecting these situations automatically can support traffic safety analysis and intelligent transportation systems.

This project investigates a video-based deep learning pipeline that can:

- Analyze traffic video sequences
- Learn spatiotemporal representations of vehicle interactions
- Detect critical/accident-related events
- Classify video segments using deep learning
- Visualize regions that influence model predictions
- Generate interpretable explanations of model decisions

The current implementation uses the **I24 highway dataset** as the primary data source.

---

## Pipeline

```text
Traffic Video
     │
     ▼
Data Preprocessing
     │
     ▼
Video Sampling / Frame Extraction
     │
     ▼
VideoMAE
     │
     ▼
Spatiotemporal Feature Extraction
     │
     ▼
Binary Classification
     │
     ├──────────────► Prediction
     │
     ▼
Grad-CAM / Attention Visualization
     │
     ▼
Important Region / Frame Analysis
     │
     ▼
LLM-based Explanation

```

---

## Core Components

### 1. Video Preprocessing

Traffic videos are processed into suitable frame sequences for model input.

Current preprocessing includes:

- Video/frame sampling
- Frame resizing
- Normalization
- Dataset organization
- Train/validation/test splitting

### 2. VideoMAE

**VideoMAE** is used as the primary video representation model.

It provides spatiotemporal feature representations by learning from sequences of video frames rather than treating individual frames independently.

### 3. Binary Classification

The extracted video representations are passed to a classification head for distinguishing between the target classes.

The classification stage is being developed and evaluated using standard machine-learning metrics.

### 4. Explainable AI

To understand why the model produces a particular prediction, the project incorporates visual explanation techniques such as:

- Grad-CAM
- Attention visualization
- Important-frame analysis

These visualizations help identify the portions of a traffic scene that contribute to the prediction.

### 5. LLM-based Explanation

The final stage is intended to convert model outputs and visual evidence into human-readable explanations.

```text
Model Prediction
       +
Visual Evidence
       +
Relevant Traffic Context
       │
       ▼
LLM Explanation

```

This is intended to make the system easier to interpret rather than relying only on a numerical classification output.

---

## Dataset

### I24 Dataset

The project currently uses the **I24 highway dataset** for traffic-scene analysis.

The dataset provides real-world highway traffic data suitable for studying vehicle interactions, traffic behavior, and safety-related scenarios.

> Dataset files are **not included in this repository** unless their redistribution is explicitly permitted by the dataset license.

Please refer to the original dataset documentation for access conditions, citation requirements, and licensing information.

---

## Technologies

| ComponentTechnology |                                    |
| ------------------- | ---------------------------------- |
| Programming         | Python                             |
| Deep Learning       | PyTorch                            |
| Video Model         | VideoMAE                           |
| Explainability      | Grad-CAM / Attention Visualization |
| Data Processing     | NumPy, Pandas, OpenCV              |
| ML Evaluation       | Scikit-learn                       |
| Experimentation     | Kaggle / GPU                       |
| Version Control     | Git + GitHub                       |

---

## Project Structure

```text
cognitive-ev-detection/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── experiments/
│
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── explainability/
│
├── configs/
│
├── outputs/
│   ├── checkpoints/
│   ├── predictions/
│   └── visualizations/
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore

```

The repository structure will evolve as the experimental pipeline develops.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/cognitive-ev-detection.git
cd cognitive-ev-detection

```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate

```

Install dependencies:

```bash
pip install -r requirements.txt

```

---

## Usage

The pipeline is currently under active development.

The general workflow is:

```bash
# 1. Prepare the dataset
python src/preprocessing/prepare_data.py

# 2. Extract/process video features
python src/models/extract_features.py

# 3. Train the classifier
python src/training/train.py

# 4. Evaluate the model
python src/evaluation/evaluate.py

# 5. Generate explanations
python src/explainability/visualize.py

```

> Script names and commands will be updated as the implementation is finalized.

---

## Evaluation

The model will be evaluated using metrics appropriate for binary classification, including:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

For safety-related event detection, particular attention will be given to **false negatives and recall**, since missing a critical event can have different implications from generating a false alert.

---

## Explainability

A major focus of this project is moving beyond:

```text
"This video is classified as an accident."

```

toward:

```text
"This video is classified as a critical event because
the model focused on specific vehicle interactions
and spatial regions within the traffic scene."

```

The explainability pipeline is intended to connect the model's prediction with observable visual evidence.

---

## Research Direction

The project is being developed as an experimental research framework exploring the combination of:

- Video transformers
- Traffic-scene understanding
- Emergency vehicle detection
- Accident/critical-event classification
- Attention-based interpretation
- Explainable AI
- Large Language Models

Future experiments may investigate model architectures, temporal representations, attention mechanisms, classification strategies, and explanation quality.

---

## Roadmap

-  Initialize project repository
-  Select I24 dataset
-  Finalize dataset preprocessing pipeline
-  Establish baseline model
-  Integrate VideoMAE
-  Train binary classifier
-  Establish evaluation pipeline
-  Implement Grad-CAM visualization
-  Analyze model attention
-  Develop LLM explanation module
-  Perform ablation experiments
-  Compare model configurations
-  Document experimental results
-  Prepare research publication

---

## Disclaimer

This project is developed for **research and educational purposes**.

The system is an experimental machine-learning framework and should not be treated as a certified safety-critical system or used as the sole basis for real-world traffic, emergency-response, or autonomous-driving decisions.

---

## License

This project is released under the **MIT License**.

See `LICENSE` for details.

---

## Citation

If this project contributes to your research, a citation format will be added once the associated research work is finalized.

```text
Cognitive EV Detection
AI-based cognitive emergency vehicle detection
and risk analysis using explainable video understanding.

```

---

## Author

**Sreegiri**

Computer Science Engineering — Cybersecurity

Research Project — Cognitive EV Detection
