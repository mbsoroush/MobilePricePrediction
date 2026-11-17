# 📱 Mobile Price Prediction

A supervised machine learning project that predicts the **price range of a mobile phone** (low, medium, high, or very high cost) from its hardware specifications. It explores which features drive price, then trains and tunes Decision Tree, Random Forest, and SVM classifiers and compares them with ensemble methods (Voting, Bagging, AdaBoost, Gradient Boosting, XGBoost). The best model, a tuned linear **SVM, reaches 97.5% accuracy**. It is then used to predict the price range of 1,000 unlabeled phones.

![Python](https://img.shields.io/badge/Language-Python%203.10-3776AB?style=flat-square)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?style=flat-square)
![scikit-learn](https://img.shields.io/badge/Library-scikit--learn-F7931E?style=flat-square)
![XGBoost](https://img.shields.io/badge/Library-XGBoost-189AB4?style=flat-square)
![Task](https://img.shields.io/badge/Task-Multiclass%20Classification-6A5ACD?style=flat-square)

## 📦 Dataset

The data is hosted on Kaggle and is not stored in this repository:

👉 [Mobile Price Range on Kaggle](https://www.kaggle.com/datasets/mbsoroush/mobile-price-range)

| File | Rows | Columns | Notes |
|------|:----:|:-------:|-------|
| `train.csv` | 2,000 | 21 | 20 features + the `price_range` target |
| `test.csv` | 1,000 | 21 | an `id` column + 20 features, **no labels** |

There are **no missing values**, and the four price classes are perfectly balanced (500 phones each).

**Target:** `price_range`: `0` Low Cost, `1` Medium Cost, `2` High Cost, `3` Very High Cost.

| Group | Features |
|-------|----------|
| **Performance** | `battery_power` (mAh), `clock_speed`, `n_cores`, `ram` (MB), `int_memory` (GB) |
| **Display** | `px_height`, `px_width`, `sc_h`, `sc_w` (screen size in cm), `touch_screen` |
| **Cameras** | `fc` (front, MP), `pc` (primary, MP) |
| **Connectivity** | `blue` (Bluetooth), `wifi`, `three_g`, `four_g`, `dual_sim` |
| **Body & battery life** | `m_dep` (depth, cm), `mobile_wt`, `talk_time` |

## 🔄 Workflow

1. **Data checks:** types, unique values, and null counts for both files.
2. **EDA:** counts and shares of the categorical features, a pair plot colored by price range, distribution and box plots for every numeric feature, count plots against price range, and a correlation heatmap.
3. **Preprocessing:** drop `id` from the test set and standardize features with `StandardScaler`, fitted on the training split.
4. **Modeling:** for each of Decision Tree, Random Forest, and SVM, pick a test size, train a baseline, then train a tuned version using `GridSearchCV` hyperparameters.
5. **Evaluation:** accuracy, macro precision, recall, F1, confusion matrices, per-class ROC curves, and 10-fold cross-validation.
6. **Ensembles:** Voting, Bagging, AdaBoost, Gradient Boosting, and XGBoost.
7. **Prediction:** the best model labels the 1,000 unlabeled phones, with names for each price range.

## 🔍 What Drives Price?

The correlation heatmap shows one dominant feature:

| Feature | Correlation with `price_range` |
|---------|:------------------------------:|
| **`ram`** | **0.92** |
| `battery_power` | 0.20 |
| `px_width` | 0.17 |
| `px_height` | 0.15 |
| all other features | close to 0 |

Among the features themselves, the strongest links are `fc`–`pc` (0.64), `three_g`–`four_g` (0.58), `px_height`–`px_width` (0.51), and `sc_h`–`sc_w` (0.51).

## 📊 Results

Test accuracy and macro-F1, plus 10-fold cross-validation accuracy:

| Model | Key settings | Test split | Accuracy | F1 | 10-fold CV |
|-------|--------------|:----------:|:--------:|:--:|:----------:|
| Decision Tree | `max_depth=3` | 25% | 0.752 | 0.749 | 0.755 |
| Improved Decision Tree | `entropy`, `max_depth=8` | 25% | 0.854 | 0.854 | 0.850 |
| Random Forest | defaults | 25% | 0.868 | 0.868 | 0.887 |
| Improved Random Forest | `gini`, 500 trees, `max_features='sqrt'` | 25% | 0.888 | 0.888 | 0.889 |
| SVM | RBF kernel, defaults | 20% | 0.893 | 0.893 | 0.951 |
| **Improved SVM** | **linear kernel, `C=10`** | 20% | **0.975** | **0.975** | n/a |
| Voting Classifier | soft voting over the 6 models above | 20% | 0.925 | 0.925 | 0.917 |
| Bagging | 50 trees, `max_depth=6`, `min_samples_split=4` | 20% | 0.875 | 0.875 | 0.871 |
| AdaBoost | 200 estimators, `learning_rate=0.1` | 20% | 0.840 | 0.840 | 0.823 |
| Gradient Boosting | 500 estimators, `learning_rate=0.1` | 20% | 0.918 | 0.918 | 0.904 |
| XGBoost | 900 estimators, `learning_rate=0.1` | 20% | 0.913 | 0.913 | 0.916 |

All splits are stratified. Cross-validation setups vary slightly between rows (see the notebook).

### 🏆 Best model: Improved SVM

A linear-kernel SVM (`C=10`) scored 97.5% on 400 held-out phones, with consistent performance across all four classes:

| Class | Precision | Recall | F1 |
|-------|:---------:|:------:|:--:|
| 0, Low Cost | 0.98 | 0.98 | 0.98 |
| 1, Medium Cost | 0.97 | 0.96 | 0.96 |
| 2, High Cost | 0.96 | 0.98 | 0.97 |
| 3, Very High Cost | 0.99 | 0.98 | 0.98 |

A linear kernel winning suggests the price classes are largely linearly separable in feature space, helped by how strongly `ram` tracks price. The ensembles reach about 91 to 93%, and the tree models are the weakest.

## 🔮 Predicting New Phones

The tuned SVM predicts a price range for each of the 1,000 phones in `test.csv`. The predictions are mapped to readable names (`Low Cost`, `Medium Cost`, `High Cost`, `Very High Cost`) and appended to the feature table. The test file has no labels, so these predictions can't be scored.

## 📁 Repository Structure

```
MobilePricePrediction/
├── mobile-price-prediction.ipynb   # EDA, modeling, ensembles, evaluation, prediction
└── README.md
```

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone <this-repository-url>
   cd <repository-folder>
   ```
2. Install the dependencies (the notebook was run with Python 3.10):
   ```bash
   pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost jupyter
   ```
   > The Bagging and AdaBoost cells use `base_estimator=`, which was **removed in scikit-learn 1.4**. Use `scikit-learn<1.4`, or rename it to `estimator=` on newer versions.
3. Download `train.csv` and `test.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/mbsoroush/mobile-price-range) and place them in the project folder.
4. Open the notebook and update the two paths in the data-loading cells to where you saved the files. The defaults are Kaggle paths:
   ```python
   train = pd.read_csv('path/to/train.csv')
   test = pd.read_csv('path/to/test.csv')
   ```
5. Launch Jupyter and run the cells in order:
   ```bash
   jupyter notebook mobile-price-prediction.ipynb
   ```

## 🎯 Learning Objectives

- Explore which features influence price with correlation analysis and visualization
- Compare tree-based and margin-based classifiers on a balanced multiclass problem
- Tune hyperparameters and judge the effect with test-set and cross-validated scores
- Evaluate multiclass models with confusion matrices, per-class metrics, and ROC curves
- Combine models with voting, bagging, and boosting
- Apply a trained model to new, unlabeled data
