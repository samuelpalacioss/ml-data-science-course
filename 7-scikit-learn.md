# Scikit-Learn

![Workflow](https://i.imgur.com/HfCGCnv.png)

### 1. Getting our data ready to be used with ML

**Three main things we have to do:**

1. Split the data into features and labels (Usually 'X' and 'y').
2. Filling or disregarding missing values.
3. Converting non-numerical values to numerical values (Feature encoding).

### 1. Making sure everything is numerical

**Converting strings to numbers, also categorical features:**

```python
# car_sales['Doors'].value_counts() # Different n of doors can refer to different categories

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ['Make', 'Colour', 'Doors']
one_hot = OneHotEncoder()
transformer = ColumnTransformer([('one_hot',
                                  one_hot,
                                  categorical_features)],
                                  remainder='passthrough')

transformed_X = transformer.fit_transform(X)
transformed_X
```

```python
pd.DataFrame(transformed_X).head()
```

![Transformed](https://i.imgur.com/T8xWKCR.png)
![OneHot](https://i.imgur.com/N4bkUjc.png)

### 2. What if there were missing values?

1. Fill them with some value.
2. Remove the samples with missing data altogether.

**Option 1.1 Fill missing data with Pandas:**

```python
car_sales_missing.isna().sum() # Missing values in each column
car_sales_missing['Doors'].value_counts() # Counts the number of times each unique value (3,4,5) appears in the Doors column

# Fill 'Make', 'Colour', 'Odometer (KM)' and 'Doors' column
car_sales_missing.fillna({
    'Make': 'missing',
    'Colour': 'missing',
    'Odometer (KM)': car_sales_missing['Odometer (KM)'].mean(), # Replace missing data with la media del odometer
    'Doors': 4
}, inplace=True)

```

**Option 1.2 Fill missing data with Scikit-Learn:**

**Option 2. Remove samples with missing data**

```python
car_sales_missing.dropna(inplace=True)
car_sales_missing.isna().sum() # Should return 0 in each col
```

#### [Feature scaling](https://rahul-saini.medium.com/feature-scaling-why-it-is-required-8a93df1af310)

### 3. Choosing the right model/estimator for your problem

[Choosing the right estimator](https://scikit-learn.org/stable/machine_learning_map.html)

- Sklearn refers to ML models, algorithms as estimators.
- Classification problem - predicting a category (heart disease or not)
  - We'll see `clf` (short for classifier) used as a classification estimator
- Regression problem - predicting a number (selling price of a car)

1. For **structured data**, use ensemble methods.
2. For **unstructured data** use deep learning or transfer learning.

### 4. Fit the model/algorithm on our data to make predictions

- **`X`**: features, data
- **`y`**: labels, targets

**Fitting a model/algorithm** on our data is the process of an algorithm learning the relationship between
feature variables and target (label) ones from a dataset.

### 4.1 Make predictions using a ML model

There are 2 ways to make predictiosn: `predict()` and `predict_proba()`

- Using `predict_proba()` returns the probability of each class (in this case 0 or 1) for a given sample,
  while `predict()` returns the most likely class label.

- The label returned by `predict()` is simply the class with the highest probability from the
  `predict_proba()` output.

### 5. Evaluating a machine learning model

There are 3 ways to evaluate Scikit-Learn models/estimators:

1. Estimators built-in `score()` method.
2. The `scoring` parameter.
3. Problem-specific metric functions.

![Cross validation, Scoring Param](https://i.imgur.com/8OU8VEq.png)

1. Using built-in `score()` method:

```python
# Import model/estimator
from sklearn.ensemble import RandomForestClassifier

# Setup a random seed
np.random.seed(42)

# Create features (X) and labels (y) matrix
X = heart_disease.drop('target', axis=1)
y = heart_disease['target']

# Split data into training and test sets
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate and fit model (on the training set)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# The highest value is 1.0, the lowest is 0.0
clf.score(X_test, y_test)
```

2. Using the `scoring` parameter

```python
# Import cross_val_score
from sklearn.model_selection import cross_val_score

# Import model/estimator
from sklearn.ensemble import RandomForestClassifier

# Setup a random seed
np.random.seed(42)

# Create features (X) and labels (y) matrix
X = heart_disease.drop('target', axis=1)
y = heart_disease['target']

# Split data into training and test sets
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate and fit model (on the training set)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

cross_val_score(clf, X, y) # 5 different splits of data, scoring param is default to None
```

### 5.1 Classification model evaluation metrics

1. Accuracy
2. Area under ROC Curve
3. Confusion Matrix
4. Classification report

**1. Accuracy**

```python
# Import cross_val_score
from sklearn.model_selection import cross_val_score
# Import estimator
from sklearn.ensemble import RandomForestClassifier

# Setup a random seed
np.random.seed(42)

# Create features (X) and labels (y) matrix
X = heart_disease.drop('target', axis=1)
y = heart_disease['target']

clf = RandomForestClassifier()
cross_val_score = cross_val_score(clf, X, y)

cross_val_score # [0.81967213, 0.90163934, 0.83606557, 0.78333333, 0.78333333]
np.mean(cross_val_score) # (0.8248087431693989

# Based on our features how likely is to predict the target
print(f'Heart dissease classifier cross-validated mean accuracy: {np.mean(cross_val_score) * 100:.2f}%')
```

**2. Area under the receiver operating characteristic curve (AUC/ROC)**

- Area under curve (AUC)
- ROC curve

_ROC curves_ compare a model's true positive rate (TPR) to its false positive rate (FPR).

- **True positive**: The model predicts 1 when the actual value is 1.
- **False positive**: The model predicts 1 when the actual value is 0.
- **True negative**: The model predicts 0 when the actual value is 0.
- **False negative**: The model predicts 0 when the actual value is 1.

**3. Confusion Matrix**

A confusion matrix compares a model's predicted labels against the actual labels, revealing where
the model is making errors (getting confused).

```python
from sklearn.metrics import confusion_matrix

y_preds = clf.predict(X_test)

confusion_matrix(y_test, y_preds)

# Better way to visualize this
pd.crosstab(y_test,
            y_preds,
            rownames=['Actual labels'],
            colnames=['Predicted labels'])
```

![Crosstab Confusion Matrix](https://imgur.com/JE3L7ZO.png)

**As we can see:**

- The bottom left to top right diagonal is where our model is failing
- 24 true negatives, where the predicted label is 0 and the actual label is 0
- 5 false positives, where the predicted label is 1 but the actual label is 0
- 5 false negatives, where the predicted label is 0 but the actual label is 1
- 27 true positives, where the predicted label is 1 and the actual label is 1

```python
(24 + 5 + 5 + 27), len(X_test) # (61, 61)
```

### Using scikit-learn to visualize confusion matrix

```python
# 1. Based on the estimator
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(clf, X, y);
# 2. Based on the predictions
ConfusionMatrixDisplay.from_predictions(y_true=y_test,
                                        y_pred=y_preds);
```

<img src="https://i.imgur.com/ymFn6HR.png" alt="Confusion Matrix Estimator" width="40%">
<img src="https://i.imgur.com/l4TFUVM.png" alt="Confusion Matrix Estimator" width="40%">

**4. Classification Report**

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_preds))
```

![ClassificationReport](https://i.imgur.com/rw7J7pO.png)

**Classification Metrics Summary:**

- **Accuracy** is a good starting point if all classes are balanced (e.g., the same number of samples are labeled with 0 or 1).
- **Precision** and **recall** become more important when classes are imbalanced.
- If false positive predictions are worse than false negatives, aim for higher precision.
- If false negative predictions are worse than false positives, aim for higher recall.
- The **F1-score** is a combination of precision and recall.

### How to install a conda package into the current env from a Jupyter Notebook

```python
import sys
!conda install --yes --prefix {sys.prefix} seaborn #seaborn is the package i want to install on my env
```
