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

- `pd.crosstab()` is a great way to view two different columns together and compare them.

```python
# Compare Car Make with number of Doors
pd.crosstab(car_sales["Make"], car_sales["Doors"])
```

![Crosstab](https://i.imgur.com/9YsTYLr.png)

- `.groupby()` is used to compare more columns in the context of another column.

```python
car_sales.groupby(["Make"]).mean(numeric_only=True) # This will give us the avg per each car maker
```

![GroupBy](https://i.imgur.com/VpFXzBx.png)

- `.plot()` is used to visualize a column.

![Graph](https://i.imgur.com/Mm0Nkg0.png)

- `.hist()` is used to see the distribution of a column.
  ![Hist](https://i.imgur.com/Mq7FmLS.png)

### 6. Manipulating data

- **To combine two or more series in a DataFrame**:

```python

pd.DataFrame({"Foods": foods, "Price": prices}) # Assuming you have foods and prices series
```

- **To access the string value of a column**, use `.str`

```python
# Prints the Make column in lowercase
car_sales["Make"].str.lower()

# Change the Make column to lowercase
car_sales["Make"] = car_sales["Make"].str.lower()
```

> [!NOTE]
> Some functions have a parameter called `inplace`, which determines wheteher an operation modifies the
> original df or series (`True`) or returns a new modified object(`False`).

- **To fill columns with missing values**, we can use `.fillna()`

```python
# Version 1. Fill Odometer column missing values with inplace=True
car_sales_missing["Odometer"].fillna(car_sales_missing["Odometer"].mean(),
                                     inplace=True)

# Version 2. Fill the Odometer missing values to the mean reassigning the column to the filled version
car_sales_missing["Odometer"] = car_sales_missing["Odometer"].fillna(car_sales_missing["Odometer"].mean())
```

- **To remove columns with missing values**, we can use `.dropna()`

```python
# Remove missing data
car_sales_missing.dropna()
```

- **To create data**, like creating a column, we have many options:

```python
# Create a column from a pandas Series
seats_column = pd.Series([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
car_sales["Seats"] = seats_column

# Create a column from a Python list
engine_sizes = [1.3, 2.0, 3.0, 4.2, 1.6, 1, 2.0, 2.3, 2.0, 3.0]
car_sales["Engine Size"] = engine_sizes

# Create a column from other columns
car_sales["Price per KM"] = car_sales["Price"] / car_sales["Odometer (KM)"]
car_sales
```

- **To set a column to a single value:**

```python
# All column to 1 value (number of wheels)
car_sales["Number of wheels"] = 4
car_sales
```

- **To remove a column**, we can use `.drop('COLUMN_NAME', axis=1)`

```python
# Drop the Price per KM column
car_sales = car_sales.drop("Price per KM", axis=1) # columns live on axis 1
car_sales
```

- **To take a random sample from a DataFrame**, use `.sample(frac=n)`, where n is the fraction of rows to
  sample:

```python
car_sales_sampled = car_sales.sample(frac=1) # Sample 100% of rows
car_sales_sampled = car_sales.sample(frac=0.5) # Sample 50% of rows
car_sales_sampled = car_sales.sample(frac=0.01) # Sample 1% of rows
```

> [!NOTE]
> Notice how the columns remain intact but their order is mixed
> ![Frac](https://i.imgur.com/i47JqsO.png)

- **To reset indexes order**, use `.reset_index()`

```python
# Reset the indexes of car_sales_sampled
car_sales_sampled.reset_index()
```

- **To apply a function to a column**, use `.apply()`

```python
# Change the Odometer values from kilometres to miles
car_sales["Odometer (KM)"] = car_sales["Odometer (KM)"].apply(lambda x: x / 1.6)
```
