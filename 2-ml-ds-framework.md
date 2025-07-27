# Machine Learning and Data Science Framework

### ML Framework

1.  **Problem definition:** What are we trying to solve? We need to identify the problem type (supervised or
    unsupervised) and the task (classification or regression).

2.  **Data:** What of kind of data do we have? Structured or Unstructured?

3.  **Evaluation:** What defines success for us? What's the required performance level for practicality?

4.  **Features:** What do we know already about or data? What patterns we can predict?

5.  **Modelling:** Based on our problem and data, what model should we use?

6.  **Experimentation:** How could we improve/what we can try next?

![Machine Learning Framework](https://i.imgur.com/D8NHy43.png)

[Framework](https://www.mrdbourke.com/a-6-step-field-guide-for-building-machine-learning-projects/)

### Types of Evaluation Metrics

| Classification | Regression                     | Recommendation |
| -------------- | ------------------------------ | -------------- |
| Accuracy       | Mean absolute error (MAE)      | Precision at K |
| Precision      | Mean squared error(MSE)        |                |
| Recall         | Root mean squared error (RMSE) |                |

### Features In Data

We want to use the feature variables (numerical, categorical) to predict the target variables (what we are
looking for).

## Modelling

- Based on our problem and data, what model should we use?

### 1. Splitting Data (3 sets)

- Since we aim to use ML models for future predictions, it’s crucial to test how well they perform in
  real-world conditions. To do this we split data in 3 different sets:

1. Training set (70%): To train our model.
2. Validation set (10-15%): To tune our model.
3. Test set (10-15%): To test and compare our different models.

> We can think of these sets as:
>
> 1. Training set (Course material)
> 2. Practice exam (Validation test)
> 3. Final exam (Test set)

### 2. Picking the Model

- **Structured data:** Decision trees (Random forests, CatBoost and XGBoost)
- **Unstructured data:** Deep Learning (Neural Networks) and Transfer learning

### 3. Tuning a model (Improving performance)

- ML Models have hyperparameters _we can adjust_. A model first results aren't its last. Tuning can take place
  on training or validation data sets.
  - Random forest: We can adjust the number of trees.
  - Neural networks: We can adjust the number of layers.

### 4. Comparison

How will our model perform in the real world?

- A good model will have similar results on the training, validation and test sets.
- It’s common to see a slight drop in performance when moving from training and validation sets to the test
  set.
- Example: Training 98%, Test 96% = **good**. Training 64%, Test 47% = **underfitting**. Training 93%, Test
  99% = **overfitting**

![Comparison](https://i.imgur.com/EGyLbPg.png)
