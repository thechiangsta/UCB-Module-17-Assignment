# UCB-Module-17-Assignment

**Link to [notebook](https://github.com/thechiangsta/UCB-Module-17-Assignment/blob/master/classification_comparisons.ipynb)**

## Introduction

This project analyzes a dataset from a Portuguese bank's direct marketing campaigns conducted via phone calls between 2008 and 2013. The dataset contains 41,188 records and 20 input features covering client demographics, contact details, and macroeconomic indicators. The goal is to predict whether a client will subscribe to a term deposit (`y = 1`) following a marketing call.

The dataset presents a significant class imbalance. Approximately 89% of clients did not subscribe (`no`) and 11% did (`yes`). This imbalance heavily influences model selection, evaluation strategy, and business interpretation of results.

## Data Description

Features fall into four categories:

**Client demographics**: age, job, marital status, education, credit default status, housing loan, personal loan

**Campaign contact details**: contact type (cell vs telephone), month and day of last contact, call duration, number of contacts in current campaign

**Previous campaign history**: days since last contact (`pdays`), number of previous contacts, outcome of previous campaign (`poutcome`)

**Macroeconomic indicators**: employment variation rate, consumer price index, consumer confidence index, euribor 3-month rate, number of employees

## Data Preparation

The following steps were taken to prepare the data for modeling:

**Cleaning**: `unknown` string values were replaced with `NaN` for proper imputation. The `pdays` value of `999` (indicating no prior contact) was replaced with `NaN`. The `duration` column was dropped entirely; per the dataset documentation, call duration is unknown before a call is made and perfectly correlated with the outcome after, making it a source of data leakage in any realistic predictive model.

**Encoding**: The `education` column was ordinally encoded to preserve its natural ordering (`illiterate < basic.4y < basic.6y < basic.9y < high.school < professional.course < university.degree`). All remaining categorical columns were one-hot encoded using `pd.get_dummies` with `drop_first=True`.

**Train/test split**: Data was split 80/20 with `stratify=y` to maintain the class ratio in both sets, yielding 32,950 training and 8,238 test records.

**Imputation**: `NaN` values were imputed using median imputation via `SimpleImputer`, fitted only on the training set to prevent data leakage.

**Scaling**: All numeric features were standardized using `StandardScaler`, again fitted only on the training set. This step is critical for distance-based and gradient-based models (KNN, Logistic Regression, SVM).

## Modeling

Four classifiers were trained and evaluated. A dummy classifier predicting the majority class (`no`) at all times was used to establish a baseline.

**Baseline**: Accuracy 89%, Class 1 F1 0.00, ROC-AUC 0.50

Given the class imbalance, `class_weight="balanced"` was applied to Logistic Regression and SVM. Models were evaluated primarily on **Class 1 F1-score** and **ROC-AUC** rather than accuracy, as accuracy is misleading on imbalanced datasets.

Hyperparameter tuning was performed using `GridSearchCV` with 5-fold cross-validation and `scoring="f1"`.

## Results

### Default models

| Model | Test F1 | ROC-AUC | Train Time |
|---|---|---|---|
| Logistic Regression | 0.46 | 0.80 | 0.04s |
| KNN | 0.38 | 0.74 | 0.006s |
| Decision Tree | 0.32 | 0.62 | 0.09s |
| SVM | 0.35 | 0.70 | 72.3s |

The default Decision Tree showed severe overfitting (train accuracy 99.47%, test accuracy 84.01%), and SVM was prohibitively slow with the default RBF kernel.

### Tuned models

| Model | Best Params | Test F1 | ROC-AUC | Train Time |
|---|---|---|---|---|
| Logistic Regression | `C=100, solver=lbfgs` | 0.46 | 0.80 | 0.04s |
| KNN | `metric=manhattan, n_neighbors=5, weights=uniform` | 0.39 | 0.75 | 0.002s |
| Decision Tree | `max_depth=7, min_samples_leaf=10, min_samples_split=50` | 0.48 | 0.80 | 0.05s |
| SVM (LinearSVC) | `C=1` | 0.32 | 0.80 | 0.44s |

Tuning had the most dramatic effect on the Decision Tree. Constraining depth fixed the overfitting and pushed ROC-AUC from 0.62 to 0.80. Logistic Regression was already near it's optimum at default settings.

---

## Findings and Business Recommendation

**Recommended model: Logistic Regression**

Logistic Regression is the recommended model for the following reasons:

- **Performance**: AUC of 0.80 and recall of 0.64 on the minority class, correctly identifying 64% of likely subscribers
- **Efficiency**: Fits in 0.04 seconds, enabling frequent retraining as new campaign data becomes available
- **No overfitting**: Train and test accuracy are nearly identical (82.94% vs 83.28%), indicating stable generalization to new data
- **Interpretability**: Coefficients can be converted to odds ratios and communicated directly to business stakeholders without a statistics background

**Practical impact**

The model's primary business value is in prioritizing which clients to call. Rather than contacting all clients indiscriminately, the bank can rank clients by their predicted subscription probability and focus effort on the top segments. This reduces wasted calls on unlikely subscribers while concentrating resources on higher-value targets.

**Key drivers of subscription (from LR coefficients)**
 
Positive drivers (increase subscription likelihood):
- Successful outcome from a previous campaign (`poutcome_success`), the single strongest predictor
- Contact made in March, August, or December
- Higher consumer price index (`cons.price.idx`)
- No previous campaign contact (`poutcome_nonexistent`)
- Higher euribor 3-month rate and number of employees (`nr.employed`)
- Client is retired or a student
- Client is single
 
Negative drivers (decrease subscription likelihood):
- High employment variation rate (`emp.var.rate`), the single strongest negative predictor
- Contact via telephone rather than cellular
- Contact made in June, May, or November
- Contact made on a Monday
 
**Threshold tuning**
 
The default classification threshold of 0.50 can be adjusted depending on business priorities. Lowering the threshold increases recall (catching more subscribers at the cost of more false positives, i.e. unnecessary calls). Raising it increases precision (fewer wasted calls but more missed subscribers). The optimal threshold depends on the relative cost of a missed subscriber vs an unnecessary call. A decision for the business rather than the model.