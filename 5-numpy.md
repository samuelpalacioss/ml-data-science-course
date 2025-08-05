# Numpy

Numpy is a library used for working with numerical computing.

![Numpy Array](https://i.imgur.com/hs6oxQt.png)

#### The main datatype is `ndarray`

```python
import numpy as np
a1 = np.array([1,2,3]) # Vector
a2 = np.array([ [1,2,3],   # 2 dimensional array
       [4,5,6] ])
```

**To know the size of an array, use `.size`**

```python
    a1.size # 3
```

**To know the shape of an array, use `.shape`**

```python
    a1.shape # (3,)
```

**To create a dataframe based on a numpy array:**

```python
import pandas as pd
df = pd.DataFrame(a2)
```

## 1. Creating arrays

- `np.array()`
- `np.ones()`
- `np.zeros()`
- `np.arange()`
- `np.random.rand(5, 3)`
- `np.random.randint(10, size=3)`
- `np.random.seed()`, pseudo random numbers

**To create an array of n size filled with ones or zeros:**

```python
np.ones((2,3)) # np.zeros((2,3))
#  array([[1., 1., 1.],
#        [1., 1., 1.]])
```

**To create an array of sequential numbers with a given interval, use `arange`:**

```python
range_arr = np.arange(0,10,2) #start at 0, stop at 10, increment by 2
# array([0, 2, 4, 6, 8])
```

**To create an array with random ints in a specific interval, use `random.randint`:**

```python
random_arr = np.random.randint(0, 10, size=(3,3))
# array([[3, 0, 7],
#       [5, 1, 0],
#       [0, 0, 6]], dtype=int32)
```

## 2. Viewing arrays and matrices

**To view unique values in an array, use `unique`:**

```python
a4 = np.random.randint(2,10, size=(3,3))
# array([[5, 9, 3],
#       [8, 2, 2],
#       [6, 9, 8]], dtype=int32)

np.unique(a4) # array([2, 3, 5, 6, 8, 9], dtype=int32)
```

**To select a certain element in an array or matrix, use `[]` notation:**

```python
# a1 array([1, 2, 3])

# a2 array([[1, 2, 3],
#          [4, 5, 6]])

a1[0] # 1

a2[0] # array([1, 2, 3])
a2[0][1] # 2
```

**To get the first n values of each row, we can use slices:**

```python
a2[:2,:2]
# array([[1, 2],
#       [4, 5]])
```

## 3. Manipulating arrays

**Arithmetic:**

- `+, -, *, /, //, **`
- `np.exp()`
- `np.log()`
- `Dot product, np.dot()`

```python
a1 # array([1, 2, 3])
a2 # array([[1, 2, 3],
   #       [4, 5, 6]])

ones = np.ones(3) # array([1., 1., 1.])

a1 + ones # array([2., 3., 4.])
a1 - ones # array([0., 1., 2.])

a1 * a2  #  array([[ 1,  4,  9],
         #         [ 4, 10, 18]])

a1 / ones # array([1., 2., 3.])

a1 ** 2 # array([1, 4, 9])
```

- Use Python's methods `sum()` on Python datatypes and use NumPy's methods on NumPy arrays `np.sum()`.

**Aggregation**:

- `np.sum()`

- `np.mean()`

- `np.std()` - Standard Deviation

- `np.var()`

- `np.min()`

- `np.max()`

- `np.argmin()` - find index of minimum value

- `np.argmax()` - find index of maximum value

**Reshaping**

- `np.reshape()`

**Transposing**

- `a3.T`

**Comparison operators**

- `> , <`
- `<= , >=`
- `x != 3 , x == 3`
- `np.sum(x > 3)`

**Standard deviation** and **variance** are measures of _spread_ of data.

- Standard deviation = sqrt(variance)
- Higher variance = wider range of numbers
- Lower variance = lower range of numbers

```python
np.var(a2) # 2.9166
np.sqrt(np.var(a2)) # std 1.7078
```

**Std and var in action:**

```python
high_var_array = np.array([1,100,200,500,1000,5000])
low_var_array = np.arange(1,10,2)

np.var(high_var_array), np.var(low_var_array) # (3098511.25), (8.0) # Varianza
np.std(high_var_array), np.std(low_var_array) # (1760.258), (2.8284) # Desviacion estandar
np.mean(high_var_array), np.mean(low_var_array) # (1133.5), (5.0) # Media
```

### Reshaping and Transposing

We can change the shape of an array without alterating its data by using `.reshape()`

```python
a2    # array([[1, 2, 3],
      #        [4, 5, 6]])

a2_reshape = a2.reshape(2,3,1)
```

![Reshape](https://i.imgur.com/gaAIQ2x.png)

We can transpose a matrix by using `.T`

```python
a2.T # array([[1, 4],
     #        [2, 5],
     #        [3, 6]])
```

### Dot product and Element wise

![DotElemet](https://i.imgur.com/YK1loEI.png)

```python
np.dot(mat1,mat2) # Dot product
mat1 * mat2       # Element wise
```

![DotProduct](https://i.imgur.com/8QsFNWz.png)
![ElementWise](https://i.imgur.com/bYxV6UW.png)

### 4. Sorting arrays

- `np.sort()` - sort values in a specified dimension of an array.
- `np.argsort()` - return the indices to sort the array on a given axis.
- `np.argmax()` - return the index/indicies which gives the highest value(s) along an axis.
- `np.argmin()` - return the index/indices which gives the lowest value(s) along an axis.

### 5. Turning images into numpy arrays

Why? Because computers can use the numbers in the numpy array to find patterns

```python
from matplotlib.image import imread

panda = imread('images/numpy-panda.jpeg')
print(type(panda)) # numpy.ndarray

panda.size, panda.shape, panda.ndim # (3271680, (852, 1280, 3), 3)
```
