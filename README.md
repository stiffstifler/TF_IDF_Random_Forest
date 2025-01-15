# Multi-Label Text Classification using TF-IDF and Random Forest

## Task
Classify textual data using a multi-label approach.

## Stack
- **TF-IDF**: For text vectorization.
- **Random Forest**: For multi-label classification.

## Steps
1. **Data Loading and Normalization**:
   - Loaded data from an Excel file and cleaned text for better processing.
2. **Label Binarization and Title Vectorization**:
   - Converted text labels to binary format and vectorized titles using TF-IDF.
3. **Data Splitting**:
   - Split the data into training and testing sets with an 80/20 ratio.
4. **Model Training**:
   - Trained the Random Forest model with 100 estimators.
5. **Model Evaluation**:
   - Measured model performance using precision, recall, F1-score, and overall accuracy.

## Results
### Accuracy
- **Achieved**: **89%**
- **Comparison**: 9% higher than the Logistic Regression model.

### Classification Report

| **Label**                   | **Precision** | **Recall** | **F1-Score** | **Support** |
|-----------------------------|---------------|------------|--------------|-------------|
| **Chief Officer**           | 0.96          | 0.60       | 0.74         | 40          |
| **Director**                | 0.93          | 0.93       | 0.93         | 97          |
| **Individual Contributor/Staff** | 0.97   | 0.98       | 0.97         | 226         |
| **Manager**                 | 0.85          | 0.53       | 0.65         | 32          |
| **Owner**                   | 0.00          | 0.00       | 0.00         | 2           |
| **Vice President**          | 0.93          | 0.93       | 0.93         | 67          |
| **Micro Avg**               | 0.95          | 0.89       | 0.92         | 464         |
| **Macro Avg**               | 0.77          | 0.66       | 0.70         | 464         |
| **Weighted Avg**            | 0.94          | 0.89       | 0.91         | 464         |
| **Samples Avg**             | 0.92          | 0.91       | 0.91         | 464         |

---
### Total model accuracy: 0.8928571428571429

## Challenges
1. **Owner Label Issue**:
   - The `Owner` label consistently received a precision, recall, and F1-score of `0.00` due to a lack of examples in the dataset.
2. **Rare Class Handling**:
   - Insufficient attention to rare classes, leading to imbalanced predictions.

---

## Solutions Undertaken
1. Used NLTK for high-quality noise cleanup.
2. Added lemmatization for better text normalization.
3. Tested different parameters, such as `n_estimators` and `test_size`.

---

## Other Optimization Possibilities
1. **Class Balancing**:
   - Implement oversampling or weighted loss functions to focus on rare metrics like `Owner`.
2. **Enhanced Vectorization**:
   - Experiment with advanced vectorization techniques, such as `Word2Vec`.
3. **Model Exploration**:
   - Test alternative models, such as MLPs or transformer-based models like BERT, for better handling of text data.

---

## Installation and Usage
### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/stiffstifler/TF_IDF_Random_Forest.git
   cd TF_IDF_Random_Forest
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:
   ```bash
   python main.py
   or
   python3 main.py
   ```

## Requirements
- nltk==3.9.1
- pandas==2.2.3
- scikit-learn==1.6.1

---
## Attention
The dataset is missing for privacy reasons.
