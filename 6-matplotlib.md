# Matplotlib

Matplotlib allows us to turn our data into graphs

_Importing matplotlib:_

```python
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
```

### 1. Creating a graph (figure)

```python
x = [1,2,3,4]
y = [2,4,9,16]
fig , ax = plt.subplots()
ax.plot(x,y);
```

![Graph](https://i.imgur.com/o2ojwuV.png)

### 2. Anatomy of a matplotlib plot

![Anatomy](https://i.imgur.com/ZpMDkYj.png)

### 3. Matplotlib Workflow

```python
# 1. Import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt

# 2. Prepare data

x = [1,2,3,4]
y = [11,22,33,44]

# 3. Setup plot
fig, ax = plt.subplots(figsize=(5,5))

# 4. Plot data
ax.plot(x,y)

# 5. Customize plot
ax.set(title='Sample plot', xlabel = 'x-axis', ylabel ='y-axis')

# 6. Save and show
fig.savefig('images/sample-plot.png')
```

![Plot](https://i.imgur.com/xkuCSeR.png)

### 4. Making plots from numpy arrays

```python
# 1. Import numpy
import numpy as np

# 2. Create data
x = np.linspace(0, 10, 100)

# 3. Plot the data and create a line plot
fig, ax = plt.subplots()
ax.plot(x, x**2);

# 4. Plot the data and create a scatter plot
fig, ax = plt.subplots()
ax.scatter(x, np.exp(x));
```

![Line Plot](https://i.imgur.com/IidlazI.png)
![Scatter Plot](https://i.imgur.com/GPKzp5T.png)

### 5. Making plots from dictionaries

**Vertical Bar Plot:**

```python
# 1. Create data
nut_butter_prices = {
    'Almond butter': 10,
    'Peanut butter': 8,
    'Pistaccio butter': 15
}

# 2. Plot the data
fig, ax = plt.subplots(figsize=(4.5,4.5))

# 3. Create a vertical bar plot
ax.bar(nut_butter_prices.keys(), height=nut_butter_prices.values())

ax.set(title='Butter store',
       ylabel= 'Price ($)')
```

![Vertical Bar Plot](https://i.imgur.com/bf2em04.png)

**Horizontal Bar Plot:**

```python
# 2. Plot the data
fig, ax = plt.subplots(figsize=(4.5,4.5))

# 3. Create a horizontal bar plot
ax.barh(list(nut_butter_prices.keys()), list(nut_butter_prices.values()))

ax.set(title='Butter store',
       xlabel= 'Price ($)')
```

![Horizontal Bar Plot](https://i.imgur.com/vMZmb10.png)

### 6. Histograms

**To create a histogram:**

```python
# 1. Create data
x = np.random.randn(100)
# 2. Plot the data
fig, ax = plt.subplots()
# 3. Create a histogram
ax.hist(x);
```

![Histogram](https://i.imgur.com/Pjkwpsk.png)

### 7. Subplots

**Option 1 for subplots:**

```python
# Option 1
fig, ( (ax1, ax2), (ax3,ax4) ) = plt.subplots(nrows=2,
                                              ncols=2,
                                              figsize=(10,5))

# Plot to each different axis
ax1.plot(x, x/2); # Top left
ax2.scatter(np.random.random(10), np.random.random(10)); # Top right
ax3.bar(nut_butter_prices.keys(), height=nut_butter_prices.values()); # Bottom left
ax4.hist(np.random.random(100)); # Bottom right
```

![Subplots](https://i.imgur.com/s2O67h3.png)

**Option 2 for subplots:**

```python
# Option 2
fig, ax= plt.subplots(nrows=2,
                      ncols=2,
                      figsize=(10,5))

# Plot to each different axis
ax[0,0].plot(x, x/2); # Top left
ax[0,1].scatter(np.random.random(10), np.random.random(10)); # Top right
ax[1,0].bar(nut_butter_prices.keys(), height=nut_butter_prices.values()); # Bottom left
ax[1,1].hist(np.random.random(100));
```

Si, de un tiempo para acá bancamiga casi que no vende. Ando buscando en qué otro banco abrirme una cuenta

### Plotting from a Pandas Dataframe

```python
car_sales.plot(x="Sale date", y="Total Sales"); # The axis are integer columns in the df
```

### Which method should I use? Pyplot or Matplot Object Oriented?

- When plotting something quickly, pyplot method is ok.
- When plotting something more advanced, use Matplot OO method.
