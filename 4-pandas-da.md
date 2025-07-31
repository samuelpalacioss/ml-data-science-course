# Pandas: Data Analysis

- Pandas is used to analyze data, it helps us get our data ready for ML.

```python
import pandas as pd
```

### 1. Data Types

1.  **Series:** A serie is a one dimensional column.

```python
brands = pd.Series(["BMW", "Mazda", "Kia"])
colours = pd.Series(["Green", "Orange", "Black"])
```

![One dimensional column](https://i.imgur.com/r5IsCtn.png)

2.  **Dataframe:** A dataframe (df) is a two dimensional column. We can create a dataframe out of series.

```python
cars_data = pd.DataFrame({"Car make": brands, "Colour": colours})
cars_data # This will print the following
```

![Car data](https://i.imgur.com/f8hv6y0.png)

### 2. Importing data

> [!NOTE]
> Rather than creating series and dataframe's from scratch, what we'll usually be doing is importing data from
> a CSV spreadsheet

```python
car_sales = pd.read_csv("car-sales.csv") # Also accepts urls
```

### Anatomy of a DataFrame

![Anatomy Of a DataFrame](https://i.imgur.com/XdUkldW.png)

### 3. Exporting data

We can also export a DataFrame, like this:

```python
car_sales.to_csv("exported-car-sales.csv", index=False) # Do not return extra index column
```

### 4. Describing data

`.dtypes` shows us what datatype each column contains.

![DataTypes](https://i.imgur.com/N3KB2ah.png)

`.describe()` gives us a quick statistical overview of the numerical columns.

![Stats](https://i.imgur.com/XTfLXq5.png)

`.info()` shows a handful of useful information about a DataFrame such as:

- How many entries (rows) there are
- Whether there are missing values
- The datatypes of each column

![Info](https://i.imgur.com/kThqeOJ.png)

> [!NOTE]
> We can also use many statistical and mathematical methods like: `.mean(), .sum()` directly on a DataFrame or Series.

`.mean(numeric_only=True)` gives us the average of numeric columns.

![Avg](https://i.imgur.com/NCPd4Mo.png)

`.sum(numeric_only=True)` gives us the sum of every column or a specific one

```python
car_sales["Doors"].sum() # 40 (Sum of the doors column)
```

![TotalSum](https://i.imgur.com/roUqrbL.png)

### 5. Viewing and selecting data

Some useful methods to view and select data from a pandas DataFrame are:

- `Dataframe.head(n)` Displays the first n rows of a df, if n is not provided it will default to 5.
- `DataFrame.tail(n)` Displays the last n rows of a DataFrame.
- `DataFrame.loc[]` Accesses a group of rows and columns by labels or a boolean array.
- `DataFrame.iloc[]` Accesses a group of rows and columns by integer indices (e.g. car_sales.iloc[0] shows
  all the columns from index 0.
- `DataFrame.columns` Lists the column labels of the DataFrame.
- `DataFrame['A']` Selects the column named 'A' from the DataFrame.
- `DataFrame[DataFrame['A'] > 8]` Boolean indexing filters rows based on column values meeting a condition
  (e.g. all rows from column 'A' greater than 8)
