I. README 

1.	Title

Novel hybridized learning-based model for heart disease prediction using PCA and deep learning

2.	Description
This project employs a two-stage strategy for predicting heart disease using machine learning (ML).  In the first stage, principal component analysis (PCA) is applied for feature selection. Features with low standard deviation that still yield good classification performance are retained. 
In the second stage, a deep autoencoder (DAE) is applied to the PCA-selected features to extract high-level representations further. The resulting features are then used to train and evaluate various ML models. 
The project compares prediction performance across two pipelines: PCA combined with machine learning (PCA+ML), and PCA followed by DAE combined with machine learning (PCA-DAE+ML), demonstrating improvements achieved by the latter approach.

3.	Dataset Information
The dataset contains 303 records with 55 features. The dataset is labeled with the binary target class ‘Cath’.  There are no missing values in this dataset. It was preprocessed using StandardScaler.

4. Code Information
The implementation is organized into three main code files that reflect the sequential steps of the experimentation pipeline: data preprocessing, dimensionality reduction using PCA, and further refinement using a deep autoencoder. Below is a summary of each file and its respective functionality:

4.1 1.preprocessing.ipy
This script handles the initial preparation of the dataset, including:
• Label Encoding: All categorical features are converted into a numeric format using label encoding.
• Feature Scaling: The dataset is scaled using three different scalers: StandardScaler, RobustScaler, and MinMaxScaler.
• Data Export: The preprocessed datasets are saved in serialized format using joblib to be reused in subsequent scripts without repeating preprocessing steps.

4.2 2.PCA_andBest_Dim_.ipynb
This notebook performs dimensionality reduction and classical machine learning evaluations:
• PCA Reduction: The dataset is reduced to multiple feature dimensions (10, 15, 20, 25, 30, 35, 40, 45, and 50) using Principal Component Analysis (PCA).
• Classifier Evaluation: For each dimension and scaling method, nine classifiers are trained and evaluated: Logistic Regression, Random Forest, K-Nearest Neighbors, XGBoost, Decision Tree, Naive Bayes, Support Vector Machine, AdaBoost, and LightGBM.
• Hyperparameter Tuning: Best hyperparameters are selected using randomized or grid search, focusing on high accuracy with low standard deviation.
• Selection: The best result was observed using 10 PCA components in combination with StandardScaler, and this configuration was selected for further refinement.

4.3 3.PCA_andDeepAutoEncd.ipynb
This script builds on the PCA-reduced data (with 10 features) by applying deep learning:
• Deep Autoencoder: A neural network-based autoencoder is trained on the selected PCA-reduced data to refine the feature representation.
• Classifier Reevaluation: The same nine ML classifiers (with previously optimized hyperparameters) are retrained and tested on the new features produced by the autoencoder.
• Performance Comparison: Performance is compared between PCA-only and PCA+Autoencoder pipelines, showing a noticeable improvement in classification accuracy with the autoencoder refinement.

NOTE : If the code in 2.PCA_andBest_Dim_.ipynb or in 3.PCA_andDeepAutoEncd.ipynb, is not responding well, make sure to add in the same repository all those
       other files available at : https://github.com/BIZIPC/HD-PCA-DAE-Project.git

5. Usage Instructions
The following steps outline how to run the code files in the correct order to reproduce the entire workflow, from data preprocessing to final model evaluation:

Step 1: Preprocessing
Run preprocessing.ipy to:
• Encode categorical features.
• Apply feature scaling using StandardScaler, RobustScaler, and MinMaxScaler.
• Save the processed datasets as .joblib files for later reuse.
Output: Scaled datasets saved as .joblib files (e.g., scaled_standard.joblib, scaled_minmax.joblib, etc.)

Step 2: PCA-Based Evaluation
Run pca.ipy to:
• Load the preprocessed datasets.
• Apply PCA for dimensionality reduction across different target dimensions (10–50).
• Evaluate nine ML classifiers under different scalers and dimensions.
• Select the optimal configuration (best-performing model, dimension, and scaler).
Output: Evaluation reports, accuracy metrics, and identification of the best configuration (10 PCA features with StandardScaler).

Step 3: Deep Autoencoder Refinement
Run 3.PCA_andDeepAutoEncd.ipynb to:
• Load the best PCA-reduced dataset (with 10 features).
• Train a deep autoencoder for feature refinement.
• Retrain and evaluate the same classifiers using the autoencoder-refined features.
• Compare the performance to PCA-only results.
Output: Final evaluation metrics and identification of the top-performing model using the autoencoder-refined features.

6.	Requirements
To successfully run the code files (1.preprocessing.ipy, 2.PCA_andBest_Dim_.ipynb, and 3.PCA_andDeepAutoEncd.ipynb), the following software and Python packages must be installed and properly configured.
Install first the package watex to get the advantages of the ML utilities. 
! pip install watex

A. Software Requirements
• Python ≥ 3.7
• Jupyter Notebook (for .ipy files, if run interactively). 
• An IDE such as VS Code, JupyterLab, or simply the command line terminal (for .py files)
• Joblib (for saving/loading preprocessed datasets)
• Matplotlib / Seaborn (for visualization, if used in evaluation)
• Scikit-learn (for ML models, preprocessing, PCA, evaluation)
• TensorFlow or Keras (for building and training the deep autoencoder)
• NumPy / Pandas (for data handling and processing)

B. Python Package Dependencies
You can install all necessary dependencies using the following command:
pip install numpy pandas scikit-learn joblib matplotlib seaborn tensorflow

C. Hardware Recommendations
• A system with at least 16 GB RAM
• For faster training of the deep autoencoder: GPU support (if using TensorFlow-GPU)

7.	Methodology 
This project follows a structured three-stage ML pipeline to perform feature preprocessing, dimensionality reduction, and model evaluation for binary classification of heart disease data. The methodology is divided into three main parts, each implemented in a separate code file.

Step 1: Data Preprocessing (preprocessing.ipy)
• Objective: Prepare the raw dataset for modeling.
• Process:
o Label Encoding is applied to categorical features to convert them into a numerical format.
o Feature Scaling is performed using three scalers: StandardScaler, RobustScaler, and MinMaxScaler.
o The preprocessed datasets are saved using joblib for efficient reuse.

Step 2: PCA-Based Dimensionality Reduction and Classifier Evaluation (2.PCA_andBest_Dim_.ipynb)
• Objective: Identify the optimal feature subset and model configuration for classification.
• Process:
o Dimensionality reduction using Principal Component Analysis (PCA) with various feature dimensions: 10, 15, 20, 25, 30, 35, 40, 45, and 50.
o Nine classifiers are trained and evaluated: Logistic Regression, Random Forest, K-Nearest Neighbors, XGBoost, Decision Tree, Naive Bayes, Support Vector Machine, AdaBoost, and LightGBM.
o Multiple scaling techniques (StandardScaler, RobustScaler, MinMaxScaler, and Normalizer) are tested in combination with classifiers.
o Hyperparameter tuning is performed using appropriate search methods.
o The combination of StandardScaler with 10 PCA components yields the best performance (highest accuracy and lowest standard deviation).

Step 3: Deep Autoencoder Refinement (3.PCA_andDeepAutoEncd.ipynb)
• Objective: Further refine the PCA-selected features using deep learning for improved performance.
• Process:
o A deep autoencoder is trained on the 10 PCA-selected features to capture nonlinear relationships and compress the data.
o The encoded (compressed) data is used to re-train the same nine classifiers with the previously optimized hyperparameters.
o Model performance is re-evaluated. The autoencoder-refined data improves classification accuracy compared to PCA-only results.
o The classifier with the highest performance is selected as the final model.

8.	Citations 
Dataset
This project is based on the Z-Alizadeh Sani Dataset, which focuses on heart disease diagnosis and is publicly available from the UCI Machine Learning Repository:
Alizadehsani, R., Roshanzamir, M., & Sani, Z. (2017).
Z-Alizadeh Sani Dataset.
UCI Machine Learning Repository.
https://archive.ics.uci.edu/ml/datasets/Z-Alizadeh+Sani

II. Materials & Methods

a. Computing Infrastructure
The experiments were conducted using a personal computing environment with the following specifications:
• Processor: Intel Core i7 (10th Gen) @ 2.60 GHz
• RAM: 16 GB
• Operating System: Windows 10 (64-bit)
• GPU: NVIDIA GeForce GTX 1650 (optional, not used for training)
• Python Version: 3.7
• Development Environment: Jupyter Notebook (via Anaconda distribution)
All ML and DL experiments were executed locally using the CPU. The code was developed and run using the JupyterLab interface, with key libraries such as:
• scikit-learn for classical ML models and preprocessing
• xgboost, lightgbm for gradient boosting
• tensorflow and keras for implementing the Deep Autoencoder
• joblib for model serialization
• matplotlib, seaborn, and pandas for visualization and data handling
This infrastructure was sufficient to handle the dataset and perform all training, validation, and evaluation tasks without requiring GPU acceleration.

b. Evaluation Method
The evaluation process followed a structured, multi-phase design incorporating both classical and deep learning approaches, and included robust validation techniques to ensure model generalizability:

1.	Preprocessing and Scaling
All categorical features were label-encoded, and numerical features were scaled using three different normalization techniques — StandardScaler, RobustScaler, and MinMaxScaler. The preprocessed datasets were saved using joblib for efficient reuse.

2.	Dimensionality Reduction via PCA
Principal Component Analysis (PCA) was applied to reduce feature dimensionality. Datasets were transformed across multiple target dimensions: 10, 15, 20, 25, 30, 35, 40, 45, and 50. For each reduced dataset, nine machine learning classifiers were trained and tested. Evaluation was done using:
o 5-fold cross-validation
o 10-fold cross-validation
o A fixed 70/30 train-test split
Performance metrics were compared across these evaluation settings. The PCA configuration with 10 components using StandardScaler was selected for its superior accuracy and low variance.

3.	Deep Autoencoder Refinement
The 10-dimensional PCA-transformed dataset was passed through a Deep Autoencoder to obtain a more compact and informative latent representation. The encoder output was used to retrain the same nine classifiers, using their previously optimized hyperparameters.

4.	Model Comparison and Selection
Classifier performance was evaluated before and after applying the autoencoder. The comparison aimed to assess performance improvement due to deep feature refinement. The model with the best accuracy after refinement was selected as the final recommended model.

5.	Validation Strategy
All models were validated using:
o 5-fold cross-validation
o 10-fold cross-validation
o A fixed 70% training / 30% testing split
This multi-scheme evaluation helped verify that the selected models are stable and generalize well across different data partitions.

c. Assessment Metrics
To comprehensively evaluate model performance, the following standard classification metrics were used:
Accuracy,  Precision, Recall, and F1-Score.  AUC-ROC (Area Under the Receiver Operating Characteristic Curve). This metric measures the classifier’s ability to distinguish between classes. A higher AUC indicates better model performance across different classification thresholds. Standard Deviation of Accuracy :  For cross-validation experiments, the standard deviation of accuracy across folds was computed to assess model stability.

III. Conclusion Section
o The dataset is relatively small, which may limit generalization.
o Real-world clinical validation is needed to reinforce or confirm model effectiveness.
o Only PCA and DeepAutoencoder were explored.
