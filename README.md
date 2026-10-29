# Adaptive Cognitive Pressure Field (A-CPF) + Koopman + DFCG

## 1. Overview

The uploaded Python code implements an end-to-end machine learning pipeline for estimating a driver's **Cognitive Pressure Field (CPF)** using the ExpertDrivingDataset. It combines multimodal driving data, temporal neural networks, latent-space dynamics, and a dynamic feature-coupling graph.

The main purpose is to estimate a normalized cognitive-pressure score from driving-related signals and compare the predicted scores of expert and novice drivers.

```text
Dataset acquisition and loading
            |
            v
Data synchronization and cleaning
            |
            v
Feature preparation and CPF target construction
            |
            v
+----------------+----------------+----------------+
|      ACPF      |     Koopman    |      DFCG      |
| Pressure       | Latent dynamics| Dynamic feature|
| prediction     | and forecasting| coupling graph|
+----------------+----------------+----------------+
            |
            v
Joint optimization with combined loss
            |
            v
Five-fold subject-level cross-validation
            |
            v
Final model evaluation and expert/novice analysis
```

**Important distinction:** this code estimates driver cognitive pressure. It does not directly detect an emergency vehicle, calculate emergency-vehicle arrival time, or predict a traffic collision. Those capabilities would require additional inputs, target variables, and prediction modules.

## 2. Algorithm I — Dataset Acquisition and Initialization

**Relevant code:** Sections 1–5, approximately lines 64–447.

### Objective

Prepare the execution environment, locate the dataset, initialize reproducibility settings, define model parameters, and establish output directories.

### Detailed procedure

1. Check whether required Python packages are installed.
2. Install missing packages, including PyTorch, MNE, NeuroKit2, scikit-learn, SciPy, and PyArrow.
3. Set the random seed to 42 for Python, NumPy, and PyTorch.
4. Detect whether the code is running on Kaggle or a local machine.
5. Search for the dataset in the configured path or expected Kaggle input directories.
6. If the dataset is missing in Kaggle, clone the configured GitHub repository and copy its `Dataset` directory.
7. Define the subjects, driving scenarios, feature extraction settings, neural-network dimensions, and training parameters.
8. Create directories for synchronized data, processed data, model checkpoints, logs, figures, and results.

### Default configuration

| Parameter | Value |
|---|---:|
| Expected subjects | 20 |
| Expert subjects | 10 |
| Novice subjects | 10 |
| Common sampling frequency | 100 Hz |
| Sequence length | 30 samples |
| Koopman latent dimension | 24 |
| ACPF hidden dimension | 256 |
| Batch size | 64 |
| Cross-validation folds | 5 |
| Cross-validation epochs | 30 |
| Final training epochs | 60 |
| Initial configured learning rate | `3e-4` |

These are configured values, not proof that all subjects or all epochs will actually be processed. Missing data, early stopping, or failed folds can reduce the effective counts.

## 3. Algorithm II — Multimodal Data Synchronization and Preprocessing

**Relevant code:** `DataSynchronizer`, Section 6, approximately lines 451–686.

### Objective

Combine heterogeneous driving and physiological signals into a common time-indexed dataset for each subject and scenario.

The source data may have different sampling rates and timestamps. For example, EEG may be recorded at a different rate from CAN bus data, while eye-tracking and physiological measurements may contain missing observations.

### Detailed procedure

1. **Identify the subject type.** Subjects beginning with `E` are treated as expert drivers; subjects beginning with `N` are treated as novice drivers.
2. **Load scenario timestamps.** Read EDA and BVP time-division files to determine the start and end of each driving scenario. If no matching scenario is found, use the available fallback time window.
3. **Load the available modalities.** Attempt to load EEG, EDA, BVP, eye tracking, CAN bus, and traffic recorder data.
4. **Extract the scenario segment.** Select measurements that belong to the scenario's time interval.
5. **Clean the timestamps.** Convert timestamps to numeric values, remove duplicate timestamps, sort observations, interpolate missing values, and replace remaining invalid values.
6. **Choose a reference timeline.** Prefer EEG, traffic, CAN bus or EDA timestamps, in that order. If none is available, use the first available nonempty modality.
7. **Synchronize modalities.** Align each modality to the reference timeline using nearest-timestamp matching, with tolerance based on half the median reference sampling interval. Interpolate and fill remaining gaps.
8. **Rename columns by modality.** Prefix columns with identifiers such as `EEG_`, `CAN_`, `EDA_`, `BVP_`, `Traffic_` and `EyeTrack_` to reduce naming conflicts.
9. **Attach scenario-level questionnaire labels.** Load NASA-TLX, stress, DES and SAM labels when available. Match labels by subject and scenario, with fallback matching by scenario number and then subject-level average.
10. **Align columns across scenarios.** Build the union of columns found across all segments. Add missing columns as zero-valued columns before concatenation.
11. **Concatenate and save.** Combine all scenario segments, clean non-finite values, and save the subject's synchronized dataset as a Parquet file.

### Output

```text
outputs/
└── synchronized/
    ├── E01.parquet
    ├── E02.parquet
    ├── ...
    ├── N01.parquet
    └── N02.parquet
```

The actual directory depends on the configured output path and whether the code is running on Kaggle or locally.

**Algorithmic significance:** synchronization allows the model to learn from measurements that describe the same approximate driving period rather than treating unrelated timestamps as simultaneous.

### Sampling-rate limitation

The synchronization routine does not guarantee that every signal is sampled at exactly 100 Hz. The code explicitly resamples EEG to 100 Hz, while other modalities are aligned to the selected reference timeline. Consequently, the final sampling rate depends on the chosen reference modality.

## 4. Algorithm III — Feature Preparation and CPF Target Construction

**Relevant code:** Sections 7–8 and 11, approximately lines 692–805 and 1042–1253.

### 4.1 Feature extraction

The code defines three standalone feature extractors.

#### A. EEG feature extraction

EEG data are processed using overlapping windows:

- Window duration: 2 seconds.
- Step size: 1 second.
- Theta band: 4–8 Hz.
- Alpha band: 8–13 Hz.
- Beta band: 13–30 Hz.
- Gamma band: 30–45 Hz.

For each channel, the code estimates power spectral density using Welch's method and calculates relative band power, theta-to-alpha ratio, spectral entropy, and log total power. It averages channel-level results to obtain seven features per window.

#### B. CAN bus feature extraction

The extractor selects available vehicle features:

- Speed.
- Acceleration.
- Brake.
- Steering angle.
- Jerk.
- Lane deviation.

If jerk is absent but acceleration is available, it estimates jerk as the change in acceleration divided by the time interval.

#### C. Traffic feature extraction

The traffic extractor maps available column names to four canonical features:

- Traffic density.
- Vehicle count.
- Road complexity.
- Intersection complexity.

When a feature cannot be found, it assigns a placeholder value of `0.5`.

**Integration note:** although EDA, BVP and eye-tracking features are included in the configuration, their corresponding dedicated feature extractors are not integrated into the main training pipeline shown here. Their available raw columns can still enter the model as numeric features after synchronization.

### 4.2 Global feature schema

The code scans synchronized Parquet files and creates a deterministic, sorted union of numeric feature columns.

Let the global feature schema be

\[
\mathcal{F}=\{f_1,f_2,\ldots,f_D\}
\]

where \(D\) is the number of distinct numeric feature columns.

Every subject is aligned to this same feature order. Missing feature columns are filled with zero.

### 4.3 Feature normalization

For feature \(j\), the training dataset computes its mean and standard deviation:

\[
\mu_j=\frac{1}{N}\sum_{i=1}^{N}x_{ij}
\]

\[
\sigma_j=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_{ij}-\mu_j)^2}
\]

The normalized feature is

\[
\boxed{z_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j}}
\]

If a feature has near-zero standard deviation, the code substitutes `1` to avoid division by zero.

During cross-validation, the validation dataset receives the training dataset's mean and standard deviation. This is the intended procedure for preventing normalization statistics from being estimated using the validation subjects.

### 4.4 Cognitive-pressure target

The primary supervised target is a normalized cognitive-pressure score derived from NASA-TLX and stress labels.

First, each label is independently min-max normalized:

\[
T_i=\frac{TLX_i-TLX_{\min}}
{TLX_{\max}-TLX_{\min}+\epsilon}
\]

\[
S_i=\frac{Stress_i-Stress_{\min}}
{Stress_{\max}-Stress_{\min}+\epsilon}
\]

Then the code defines:

\[
\boxed{CPF_i=\frac{T_i+S_i}{2}}
\]

where:

- \(T_i\) represents normalized workload.
- \(S_i\) represents normalized stress.
- \(CPF_i\) is the target pressure score, generally between 0 and 1.

The small constant \(\epsilon=10^{-8}\) prevents division by zero.

**Critical limitation:** these are normalized questionnaire-derived labels, not direct measurements of an objectively defined physical or psychological pressure field. The code also creates random synthetic NASA-TLX labels if it cannot find a suitable label column. If stress is missing, it copies the NASA-TLX label. Training with synthetic labels would not establish that the model can accurately estimate real cognitive pressure.

## 5. Algorithm IV — Construct Temporal Training Samples

**Relevant code:** `CognitiveDataset`, approximately lines 1063–1253.

The model learns from sequences of 30 consecutive samples rather than individual rows.

Let the normalized feature matrix be

\[
X=[x_1,x_2,\ldots,x_N]
\]

For sequence length \(L=30\), the input at starting index \(i\) is

\[
\boxed{X_i=[x_i,x_{i+1},\ldots,x_{i+29}]}
\]

The corresponding pressure target is the sequence of pressure labels at those same 30 positions:

\[
P_i=[p_i,p_{i+1},\ldots,p_{i+29}]
\]

The dataset also creates future-feature targets at horizons \(h\in\{5,10,15\}\):

\[
\boxed{Y_i^{(h)}=x_{i+L+h-1}}
\]

These future targets are used by the Koopman forecasting component, not directly by the pressure prediction head.

The dataset skips invalid samples and returns:

- `features`: the 30-step feature sequence.
- `pressure`: the corresponding pressure sequence.
- `future_targets`: future feature vectors at 5-, 10- and 15-step horizons.

With a 100 Hz reference timeline, 30 samples would represent 0.3 seconds if the data are actually sampled at 100 Hz. The forecasting horizons would correspond to 0.05, 0.10 and 0.15 seconds under the same assumption. Actual timing depends on the reference timeline.

## 6. Algorithm V — Three-Model Architecture

**Relevant code:** Section 9, approximately lines 806–988.

The architecture contains three neural networks trained together:

1. ACPF — Adaptive Cognitive Pressure Field.
2. Koopman model — latent representation and temporal forecasting.
3. DFCG — Dynamic Feature Coupling Graph.

They are optimized using a combined loss function, although their outputs are not all directly connected to the final pressure prediction.

### 6.1 ACPF — Adaptive Cognitive Pressure Field

**Purpose:** predict cognitive pressure from a sequence of driving and physiological features.

```text
Input sequence: batch × 30 time steps × D features
                    |
                    v
       Input normalization and padding
                    |
                    v
            Five gating networks
                    |
                    v
         Weighted feature representation
                    |
                    v
        Projection network (hidden = 256)
                    |
                    v
             Two-layer GRU
                    |
                    v
       Residual connection + normalization
                    |
                    v
          Prediction head + sigmoid
                    |
                    v
       Pressure prediction at each step
```

#### Step 1: Divide features into five groups

If the input dimension is \(D\), the code pads the feature dimension when necessary so that it is divisible by five:

\[
D'=5\left\lceil\frac{D}{5}\right\rceil
\]

The padded input is divided into five equal-sized groups. These groups are formed by feature position, not by explicitly identifying EEG, CAN, EDA, BVP or eye-tracking modalities.

#### Step 2: Calculate gating weights

Each group is processed by a small neural network:

\[
g_m=\operatorname{MLP}_m(X_m)
\]

The five scores are converted to weights using softmax:

\[
\boxed{\alpha_m=\frac{e^{g_m}}{\sum_{j=1}^{5}e^{g_j}}}
\]

Therefore,

\[
\sum_{m=1}^{5}\alpha_m=1
\]

The code scales the corresponding feature groups:

\[
\widetilde X_m=\alpha_m X_m
\]

This allows the network to learn which feature groups are useful for pressure prediction at different time steps.

#### Step 3: Project the weighted features

A fully connected network maps the weighted input to a 256-dimensional hidden representation. Layer normalization, GELU activation and dropout are applied.

#### Step 4: Learn temporal dependencies

A two-layer GRU processes the projected sequence. The GRU maintains a hidden state that captures information from earlier time steps.

The implementation then adds the projected representation to the GRU output and applies layer normalization:

\[
H=\operatorname{LayerNorm}(H_{\text{projected}}+H_{\text{GRU}})
\]

#### Step 5: Predict cognitive pressure

A final neural-network head maps the hidden representation through a smaller hidden layer to one output per time step:

\[
\boxed{\widehat p_t=\operatorname{Sigmoid}(W H_t+b)}
\]

The sigmoid restricts the output to \((0,1)\), matching the normalized target range.

**Output:** a sequence of predicted cognitive-pressure values and five learned gating weights per time step.

### 6.2 Koopman Model — Latent Dynamics and Forecasting

**Purpose:** learn a compact representation of the input sequence and predict future feature vectors using repeated latent-space transformations.

#### Step 1: Encode the input

For each time step \(t\), the encoder transforms the original feature vector into a latent vector:

\[
z_t=E(x_t)
\]

The encoder has hidden layers of 128 and 64 units and produces a latent representation of dimension 24.

#### Step 2: Reconstruct the input

The decoder attempts to recover the original feature vector:

\[
\widehat x_t=D(z_t)
\]

The reconstruction loss measures the difference between the input and its reconstruction.

#### Step 3: Learn the Koopman operator

The model maintains a trainable matrix \(K\) of size \(24\times24\). Its intended latent-state transition is

\[
\boxed{z_{t+1}\approx Kz_t}
\]

The code applies the transpose of the stabilized matrix to row-vector representations, equivalent to this convention when expressed with column vectors.

#### Step 4: Stabilize the operator

The code computes the singular value decomposition of \(K\):

\[
K=U\Sigma V^\top
\]

Each singular value is clipped to at most 1, producing a stabilized operator:

\[
K_{\text{stable}}=U\Sigma_{\text{clipped}}V^\top
\]

This constrains its spectral norm to at most 1 and helps prevent latent states from growing excessively during repeated forecasting.

#### Step 5: Forecast future features

Starting from the last latent state of the input sequence, the model applies the stabilized operator repeatedly:

\[
z_{t+h}=K_{\text{stable}}^h z_t
\]

The decoder maps each future latent state back to feature space:

\[
\widehat x_{t+h}=D(z_{t+h})
\]

The code produces predictions at \(h=5,10,15\).

**Output:** reconstructed features, future feature predictions, and latent representations.

This is a learned latent-space forecasting model. Its accuracy and the degree to which its latent dynamics are linear must be established experimentally.

### 6.3 DFCG — Dynamic Feature Coupling Graph

**Purpose:** construct a learned graph over the 24 latent dimensions and encourage smoothness in its output over time.

Each latent dimension is treated as a graph node:

\[
N=24
\]

#### Step 1: Form node pairs

For latent state \(z_t\), construct pairs

\[
(z_{t,i},z_{t,j})
\]

for all \(i,j\in\{1,\ldots,24\}\).

#### Step 2: Calculate edge weights

Each pair passes through a small neural network:

\[
\boxed{A_{ij}=\operatorname{Sigmoid}
\left(\operatorname{MLP}(z_{t,i},z_{t,j})\right)}
\]

Each edge weight lies between 0 and 1. The model constructs a \(24\times24\) adjacency matrix at every time step.

#### Step 3: Aggregate neighboring node states

The code calculates a graph-aggregated state:

\[
u_t=z_t+A_tz_t
\]

and applies a sigmoid:

\[
\boxed{\widetilde z_t=\operatorname{Sigmoid}(u_t)}
\]

#### Step 4: Calculate graph smoothness

During training, the model penalizes abrupt changes in the graph-processed latent representation:

\[
\boxed{L_{\text{graph}}=
\operatorname{Mean}\left[
(\widetilde z_{t+1}-\widetilde z_t)^2
\right]}
\]

This encourages the graph output to change smoothly between consecutive samples.

**Output:** graph-processed latent representations and a temporal smoothness loss.

**Architectural limitation:** the graph output is not passed into the ACPF pressure prediction head. DFCG affects its graph-related training objective rather than directly modifying predicted pressure.

## 7. Algorithm VI — Combined Loss Function

**Relevant code:** `CombinedLoss`, approximately lines 989–1037.

The code jointly optimizes five loss terms.

### 7.1 Pressure prediction loss

\[
L_{\text{CPF}}=\operatorname{MSE}(\widehat P,P)
\]

This penalizes differences between predicted and target cognitive-pressure sequences.

### 7.2 Reconstruction loss

\[
L_{\text{recon}}=\operatorname{MSE}(\widehat X,X)
\]

This encourages the Koopman encoder-decoder to retain information from the original features.

### 7.3 Future forecasting loss

\[
L_{\text{koop}}=
\frac{1}{|\mathcal H|}
\sum_{h\in\mathcal H}
\operatorname{MSE}(\widehat X^{(h)},X^{(h)})
\]

Here, \(\mathcal H\) is the set of forecast horizons that have corresponding targets.

### 7.4 Graph smoothness loss

\[
L_{\text{graph}}=
\operatorname{Mean}\left[
(\widetilde Z_{t+1}-\widetilde Z_t)^2
\right]
\]

This penalizes temporal variation in the graph-processed latent states.

### 7.5 Koopman operator regularization

\[
L_{\text{reg}}=\|K\|_2
\]

The implementation uses `torch.norm(K, p=2)`, which for a matrix is its spectral (operator) norm, not its Frobenius norm.

### Final objective

\[
\boxed{
\begin{aligned}
L_{\text{total}}={}&L_{\text{CPF}}\\
&+0.1(L_{\text{recon}}+L_{\text{koop}})\\
&+0.05L_{\text{graph}}\\
&+0.01L_{\text{reg}}
\end{aligned}
}
\]

| Loss component | Weight | Main purpose |
|---|---:|---|
| Pressure prediction | 1.00 | Predict cognitive pressure |
| Reconstruction | 0.10 | Preserve feature information |
| Future forecasting | 0.10 | Learn future feature dynamics |
| Graph smoothness | 0.05 | Encourage temporal consistency |
| Koopman regularization | 0.01 | Penalize a large operator norm |

The loss is differentiable, so gradients can update the trainable parameters of the relevant networks.

**Important distinction:** joint optimization is not the same as direct architectural fusion. ACPF, Koopman and DFCG receive separate inputs from the same feature sequence. The code does not feed Koopman forecasts or DFCG outputs into ACPF's prediction head. The auxiliary models are trained alongside ACPF, but their representations do not directly contribute to its pressure prediction.

## 8. Algorithm VII — Model Training and Optimization

**Relevant code:** Section 12, approximately lines 1255–1525.

The `CognitiveTrainer` initializes all three models, an Adam optimizer, the combined loss function, and a cosine-annealing learning-rate scheduler.

### Training configuration

| Parameter | Value |
|---|---:|
| Optimizer | Adam |
| Base learning rate | \(3\times10^{-4}\) |
| Weight decay | \(10^{-4}\) |
| Warm-up | 3 epochs |
| Gradient accumulation | 4 batches |
| Gradient clipping | Maximum norm 1.0 |
| Final early-stopping patience | 10 epochs |
| Mixed precision | Enabled when CUDA is available |

### Detailed training algorithm

1. Set the three models to training mode.
2. Load a batch containing feature sequences, pressure targets and future-feature targets.
3. Replace non-finite inputs with zero.
4. Add Gaussian noise with standard deviation \(0.02\) to training features.
5. Run the feature sequence through ACPF to predict pressure.
6. Run the same sequence through Koopman to obtain reconstruction and future forecasts.
7. Process the Koopman latent states through DFCG.
8. Compute the graph smoothness loss.
9. Combine the pressure, reconstruction, forecasting, graph and regularization losses.
10. Check whether the loss is finite. Skip failed batches and abort if more than 10% of batches are invalid.
11. Divide the loss by four and accumulate gradients over four batches.
12. Check gradients for non-finite values, zero invalid gradients and clip the gradient norm.
13. Update model parameters using Adam.
14. Evaluate the model on the validation dataset after each epoch.
15. Save a checkpoint whenever validation loss improves.
16. Stop early when validation loss fails to improve for the configured patience period.

### Learning-rate schedule

During the three warm-up epochs, the learning rate increases linearly:

\[
LR_e=LR_{\text{base}}\frac{e}{3}
\]

After warm-up, the code uses `CosineAnnealingWarmRestarts` to vary the learning rate periodically.

### Model checkpointing

The best checkpoint stores:

- ACPF model parameters.
- Koopman model parameters.
- DFCG model parameters.
- Optimizer state.
- Epoch number and best validation loss.

After training, the best saved weights are reloaded before evaluation.

## 9. Algorithm VIII — Five-Fold Subject-Level Cross-Validation

**Relevant code:** Section 14, approximately lines 1590–1764.

The code uses `StratifiedGroupKFold` to evaluate how well the model generalizes to subjects not used for training in a given fold.

### Procedure

1. Assign expert drivers class 0 and novice drivers class 1.
2. Use five folds with shuffling and random seed 42.
3. Keep each subject in only one group so that it cannot appear in both training and validation within a fold.
4. Build the global feature schema.
5. Estimate feature means and standard deviations from training subjects only.
6. Apply the training feature schema and normalization statistics to validation subjects.
7. Train fresh model instances for up to 30 epochs with early stopping.
8. Restore the best checkpoint.
9. Evaluate the ACPF model on validation subjects.
10. Calculate RMSE, MAE and \(R^2\) for each fold.
11. Save individual fold metrics and training histories.
12. Aggregate the mean and standard deviation across folds.
13. Release model and data-loader resources before starting the next fold.

### Evaluation metrics

**Root Mean Squared Error (RMSE)**

\[
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\widehat y_i)^2}
\]

RMSE penalizes large prediction errors more heavily.

**Mean Absolute Error (MAE)**

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\widehat y_i|
\]

MAE gives the average absolute prediction error.

**Coefficient of determination (\(R^2\))**

\[
R^2=1-\frac{\sum_i(y_i-\widehat y_i)^2}
{\sum_i(y_i-\bar y)^2}
\]

A value closer to 1 generally indicates better predictions. A negative value means the predictions are worse than the constant mean-target baseline on that evaluation set.

The five-fold results provide a more informative generalization estimate than evaluating only on the same subjects used for training.

## 10. Algorithm IX — Final Model Training and Evaluation

**Relevant code:** Section 16, approximately lines 2126–2297.

After cross-validation, the pipeline attempts to train a final production model.

### Procedure

1. Reload the synchronized subject datasets.
2. Build the global feature schema.
3. Create the final training dataset using all available subjects.
4. Compute normalization statistics from this dataset.
5. Create a validation dataset using subjects selected from the first cross-validation fold.
6. Initialize the three models with the resulting feature dimension.
7. Train for up to 60 epochs.
8. Restore the best checkpoint.
9. Predict cognitive-pressure values on the validation loader.
10. Calculate RMSE, MAE and \(R^2\).
11. Save prediction values, evaluation metrics and training history.
12. Generate plots of actual versus predicted pressure and training versus validation loss.

### Final evaluation limitation

The final training dataset includes all subjects, including those subsequently designated for final validation. Therefore, the reported final-model holdout metrics are not an independent holdout evaluation.

The five-fold cross-validation results are the more appropriate generalization estimates, assuming the folds are constructed correctly and no other leakage occurs.

## 11. Algorithm X — Expert-versus-Novice Cognitive-Pressure Analysis

**Relevant code:** Section 15, approximately lines 1765–2125.

The code runs the trained ACPF model over all available subjects, then compares predicted pressure trajectories for expert and novice drivers.

### Procedure

1. Load each subject's synchronized dataset.
2. Select the same feature columns used during training.
3. Normalize the features using the final training dataset's statistics.
4. Generate overlapping sequences of 30 samples with a stride of one sample.
5. Run inference in batches to reduce GPU memory consumption.
6. Extract the pressure prediction at the middle time step of each sequence.
7. Associate each prediction with its subject, scenario, driver type and sequence index.
8. Aggregate predictions by driver group and scenario.
9. Calculate summary statistics and perform a Mann–Whitney U test.
10. Export predictions and generate six visualization types.

### Outputs generated

1. **Time-series comparison** — `cpf_timeseries_expert_vs_novice.png`
2. **Distribution comparison** — `cpf_distribution_expert_vs_novice.png`
3. **Box plot and significance test** — `cpf_boxplot_significance.png`
4. **Scenario-level heatmap** — `cpf_scenario_heatmap.png`
5. **Per-subject trajectories** — `cpf_per_subject_trajectories.png`
6. **Scenario-level bar chart** — `cpf_scenario_barchart.png`

The pipeline also exports:

- `cpf_all_subjects.csv`
- `cpf_group_summary.csv`

### Statistical caution

Consecutive windows overlap heavily, so their predictions are not independent observations. The Mann–Whitney U test is applied to individual window predictions rather than independent subject-level observations. This can produce artificially small p-values.

A stronger analysis would aggregate predictions at the subject level or use a statistical model that accounts for repeated measurements.

## 12. Consolidated End-to-End Pseudocode

```text
ALGORITHM: A-CPF + KOOPMAN + DFCG

INPUT:
    ExpertDrivingDataset
    Subject IDs
    Driving scenarios
    Training configuration

OUTPUT:
    Trained pressure prediction model
    Cross-validation metrics
    Pressure predictions
    Expert-versus-novice analysis

1. INITIALIZE
    Install required packages
    Set random seeds
    Locate dataset
    Load configuration
    Create output directories

2. SYNCHRONIZE DATA
    For each subject:
        Determine expert/novice category
        Load scenario timestamps
        For each scenario:
            Load available sensor and vehicle data
            Slice each modality to the scenario
            Select reference timestamps
            Align modalities to reference timeline
            Interpolate and clean missing values
            Attach questionnaire labels
            Align feature columns
        Concatenate scenario segments
        Save synchronized subject data

3. BUILD FEATURES
    Scan synchronized datasets
    Construct global numeric feature schema
    Exclude scenario and label columns
    Align all subjects to the same feature order
    Normalize features
    Construct normalized CPF targets
    Generate 30-step input sequences
    Generate future targets at horizons 5, 10, 15
    Skip invalid samples

4. PERFORM FIVE-FOLD CROSS-VALIDATION
    Stratify subjects into five folds
    For each fold:
        Split training and validation subjects
        Fit normalization on training subjects
        Transform validation subjects with training statistics
        Initialize ACPF, Koopman and DFCG
        For each epoch:
            Add training noise
            Predict CPF with ACPF
            Reconstruct features with Koopman
            Forecast future features with Koopman
            Construct dynamic latent graph with DFCG
            Calculate combined loss
            Validate finite values and gradients
            Accumulate gradients and update parameters
            Evaluate validation loss
            Save best checkpoint
            Apply early stopping when necessary
        Evaluate RMSE, MAE and R²
        Save fold metrics
        Release resources
    Summarize cross-validation results

5. TRAIN FINAL MODEL
    Build final training dataset
    Fit normalization statistics
    Initialize three models
    Train with the combined loss
    Save and restore the best checkpoint
    Evaluate and save predictions and metrics

6. ANALYZE DRIVER GROUPS
    Run ACPF inference on all subjects
    Generate overlapping sequences
    Collect CPF predictions
    Group predictions by subject and scenario
    Calculate descriptive statistics
    Perform expert-versus-novice comparison
    Generate six visualization types
    Export CSV files

7. RETURN RESULTS
    Cross-validation metrics
    CPF prediction records
    Model checkpoints
    Figures and result files
```

## 13. Critical Code Review: Issues to Resolve Before Research Publication

The algorithm is reasonably modular, but several implementation and experimental-design issues should be addressed before interpreting its results as evidence of a reliable cognitive-pressure estimator.

### 13.1 Synthetic-label fallback

If the NASA-TLX label is not detected, the code generates random labels. A model trained on those targets cannot establish real cognitive-pressure estimation performance.

**Recommended correction:** fail explicitly when genuine labels are required, rather than silently producing synthetic targets.

### 13.2 Final validation leakage

All subjects are used to build `train_ds`, and the selected validation subjects are subsequently included in `val_ds`. This makes the final holdout metrics unreliable.

**Recommended correction:** reserve independent subjects before fitting the final model, or report cross-validation as the primary generalization result.

### 13.3 Questionnaire-label matching

The code falls back to a subject-level average when a scenario-specific questionnaire label cannot be found. This may assign the same label to unrelated scenarios. It also applies scenario-level labels to every row in that scenario, even though the labels are not necessarily time-resolved.

**Recommended correction:** document the label granularity, validate matching explicitly, and avoid presenting scenario-level questionnaire scores as continuous ground-truth measurements without justification.

### 13.4 Feature grouping is not true modality fusion

ACPF splits the sorted feature vector into five equal-sized groups. It does not explicitly group all EEG, CAN, EDA, BVP and eye-tracking features by modality.

**Recommended correction:** if modality-level fusion is intended, construct explicit modality-specific feature groups and design the gating network around those groups.

### 13.5 Auxiliary networks do not directly feed ACPF

Koopman and DFCG contribute auxiliary losses, but their outputs do not enter the pressure prediction head. If the research claim is that graph relationships and Koopman dynamics directly improve CPF prediction, the architecture needs an explicit information-fusion mechanism and suitable ablation experiments.

### 13.6 Inconsistent sampling timelines

EEG is resampled to 100 Hz, while other modalities are aligned to a selected reference index. The resulting index may not correspond to a uniform 100 Hz timeline.

**Recommended correction:** verify the final sampling interval and timestamp units before interpreting forecast horizons as physical time intervals.

### 13.7 Potential implementation typo

The supplied code listing shows:

```python
aligned["scenario"] = scenariok
```

If that is the literal source code rather than a formatting artifact, `scenariok` is undefined and synchronization will fail with a `NameError`.

**Recommended correction:** verify the actual source line and correct it to the scenario variable.

### 13.8 Statistical independence and interpretation

Overlapping windows are correlated, and the expert-versus-novice significance test treats them as independent.

**Recommended correction:** use subject-level aggregation or a repeated-measures statistical method. Also, the model estimates a learned score from available labels; it does not independently prove that experts experience lower pressure than novices.

### 13.9 Additional evaluation checks

- The global feature schema is built before cross-validation from every subject's synchronized files. A strict evaluation protocol should determine the feature schema from the training partition or justify why it is fixed independently of evaluation data.
- Final training normalization and the final evaluation split must be corrected together.
- `evaluate_and_plot_cpf` uses the final model to analyze all subjects, including training subjects. Those figures are descriptive model outputs, not independent test results.
- If the final training dataset contains no usable sequences, the code needs an explicit failure path before model initialization and evaluation.
- Verify the literal source file's syntax and variable names before execution; the listing alone does not establish that the complete script runs successfully.

## 14. Final Algorithm Interpretation

The pipeline has three distinct learning objectives:

| Component | Learns | Supervision |
|---|---|---|
| ACPF | Cognitive-pressure prediction over time | Questionnaire-derived pressure target |
| Koopman | Feature reconstruction and future feature dynamics | Original and future feature vectors |
| DFCG | Latent feature coupling and temporal smoothness | Auxiliary smoothness objective |

The central prediction task is regression: estimating a normalized pressure score at each time step. Koopman and DFCG are auxiliary learning components intended to model temporal dynamics and latent relationships.

**Key research distinction:** the code currently implements joint training of three separate branches, rather than a fully integrated model in which dynamic graph information and Koopman forecasts directly influence pressure prediction. That distinction should be resolved before making claims about the proposed architecture's novelty or effectiveness.
