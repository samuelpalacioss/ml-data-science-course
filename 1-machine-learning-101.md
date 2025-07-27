# Machine Learning 101

## What is machine learning?

Machine learning is the process of using computers to detect patterns in massive datasets and then make
predictions based on what the computer learns from those patterns.

Machine learning is good for big data (structured/unstructured data), we give this data to machines so that
they can make business decisions, even better than humans

### Types of Machine Learning

- **Supervised Learning:** The data we receive is already categorized or labeled. We can perform tasks such
  as:
  - **Classification:** Predicting a category/label, e.g., apple or pear. Two options = binary; More than two
    = multiclass.
  - **Regression:** Predicting a numerical output, such as predicting stock prices.
- **Unsupervised Learning:** The data we receive is not categorized, and the algorithm must find patterns or
  structures within it. Key tasks include:
  - **Clustering:** Grouping similar data points together. We give the machine a bunch of data and it creates
    groups.
  - **Association Rule Learning:** Associating different things to predict relationships between variables,
    e.g. what a customer perhaps _might_ buy in the future.
- **Transfer Learning**: Reusing a model pre-trained on a large dataset for a new, related task.
- **Reinforcement Learning:** Teaching machines through trial and error, e.g. a bot that plays chess.

> [!NOTE]  
> **Supervised Learning** ✅: I know my **inputs** and **outputs**.  
> **Unsupervised Learning** 🔍: I only know the **inputs**, not the outputs.  
> **Transfer Learning** 🔄: My problem may be similar to something else.
