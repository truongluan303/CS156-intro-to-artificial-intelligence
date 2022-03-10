# KNN Function Implementation

### Import the Libraries Needed


```python
try:
    # try import the required libraries
    import matplotlib
    import pandas
    import numpy
    import sklearn
except:
    # install the required libraries if import fails
    import subprocess
    from sys import executable as exe
    subprocess.check_call([
        exe, "-m", "pip", "install", "matplotlib", 
        "pandas", "numpy", "sklearn"
    ])
finally:
    # import libraries as short names
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
```


```python
np.random.seed(42)
```

### 2D Data


```python
n = 100

x1 = np.random.normal(loc=-2.0, scale=2.0, size=n//2)
x2 = np.random.normal(loc=0.0, scale=1.0, size=n//2)

y1 = np.random.normal(loc=2.0, scale=2.0, size=n//2)
y2 = np.random.normal(loc=0.0, scale=1.0, size=n//2)

x = np.concatenate((x1, y1), axis=0)
y = np.concatenate((x2, y2), axis=0)

l1 = [0] * (n // 2)
l2 = [1] * (n // 2)
labels = l1 + l2

print(labels)

data2d = pd.DataFrame(
    {'x': x, 'y': y, 'class': labels},
    columns=['x', 'y', 'class']
)
data2d.head()
```

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.006572</td>
      <td>0.324084</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.276529</td>
      <td>-0.385082</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.704623</td>
      <td>-0.676922</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.046060</td>
      <td>0.611676</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-2.468307</td>
      <td>1.031000</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

</div>




```python
train_x2d, test_x2d, train_y2d, test_y2d = train_test_split(
    data2d, labels, test_size=0.2, random_state=0
)
```

### 3D Data


```python
n = 1000

x1 = np.random.normal(loc=0.0, scale=3.0, size=n//4)
x2 = np.random.normal(loc=3.0, scale=1.0, size=n//4)
x3 = np.random.normal(loc=-1.0, scale=1.0, size=n//4)

y1 = np.random.normal(loc=0.0, scale=3.0, size=n//4)
y2 = np.random.normal(loc=1.0, scale=2.0, size=n//4)
y3 = np.random.normal(loc=1.0, scale=1.0, size=n//4)

z1 = np.random.normal(loc=0.0, scale=3.0, size=n//4)
z2 = np.random.normal(loc=3.0, scale=1.0, size=n//4)
z3 = np.random.normal(loc=4.0, scale=1.0, size=n//4)

w1 = np.random.normal(loc=0.0, scale=3.0, size=n//4)
w2 = np.random.normal(loc=5.0, scale=4.0, size=n//4)
w3 = np.random.normal(loc=-3.0, scale=1.0, size=n//4)

x = np.concatenate((x1, y1, z1, w1), axis=0)
y = np.concatenate((x2, y2, z2, w2), axis=0)
z = np.concatenate((x3, y3, z3, w3), axis=0)

l1 = [0] * (n // 4)
l2 = [1] * (n // 4)
l3 = [2] * (n // 4)
l4 = [3] * (n // 4)
labels = l1 + l2 + l3 + l4
print(labels)

data3d = pd.DataFrame(
    {'x': x, 'y': y, 'z': z, 'class': labels},
    columns=['x', 'y', 'z', 'class']
)
data3d.head()
```

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.073362</td>
      <td>2.937321</td>
      <td>-1.522723</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.682354</td>
      <td>3.955142</td>
      <td>0.049009</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.249154</td>
      <td>2.014274</td>
      <td>-1.704344</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.161406</td>
      <td>3.504047</td>
      <td>-2.408461</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-4.133008</td>
      <td>2.469742</td>
      <td>-2.556629</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

</div>




```python
train_x3d, test_x3d, train_y3d, test_y3d = train_test_split(
    data3d, labels, test_size=0.2, random_state=0
)
```

### Helpers for K-Nearest Neighbors Function


```python
def calculate_euclidean_distance(a1, a2) -> float:
    return np.sqrt(np.sum(np.square(a1 - a2)))

def make_prediction(neighbors):
    votes = dict()
    for neigh in neighbors:
        n = neigh[-1]
        if n not in votes:
            votes[n] = 0
        votes[n] += 1
    votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    return votes[0][0]
```

### K-Nearest Neighbors Function (KNN):


```python
def knn(observation, ref_data, k) -> list:
    distances = list()
    for i in range(len(ref_data)):
        d = calculate_euclidean_distance(observation[:-1], ref_data.iloc[i, :-1])
        distances.append((ref_data.iloc[i], d))
    distances.sort(key=lambda x: x[1])
    neighbors = list()
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors
```

### Function to Calculate The Accuracy of The Prediction


```python
def calculate_accuracy(expect, prediction):
    match = 0
    for a, b in zip(expect, prediction):
        if a == b:
            match += 1
        print(f"Expect: {a}\t\tPrediction: {b}")
    accuracy = 100.0 * match / float(len(expect))
    print(f"Accuracy: {accuracy}%")
```

### Function to Test


```python
def test(test_x, train_x, test_y, train_y):
    predictions = list()
    for i in range(len(test_x)):
        neighbors = knn(test_x.iloc[i], train_x, 3)
        prediction = make_prediction(neighbors)
        predictions.append(prediction)
    calculate_accuracy(test_y, predictions)

    plt.subplot(1, 2, 1)
    plt.scatter(train_x.iloc[:,0], train_x.iloc[:,1], s=25, c=train_y, marker=".")
    plt.scatter(test_x.iloc[:,0], test_x.iloc[:,1], s=50, c=test_y, marker="v")
    plt.title("Actual labels")
    plt.subplot(1, 2, 2)
    plt.scatter(train_x.iloc[:,0], train_x.iloc[:,1], s=25, c=train_y, marker=".")
    plt.scatter(test_x.iloc[:,0], test_x.iloc[:,1], s=50, c=predictions, marker="v")
    plt.title("Predicted labels")
    plt.tight_layout()
    plt.show()
```

### Test on 2D Data Set


```python
print("Running test on 2D data...")
test(test_x2d, train_x2d, test_y2d, train_y2d)
```

    Running test on 2D data...
    Expect: 0		Prediction: 0.0
    Expect: 1		Prediction: 0.0
    Expect: 0		Prediction: 0.0
    Expect: 1		Prediction: 1.0
    Expect: 1		Prediction: 1.0
    Expect: 1		Prediction: 1.0
    Expect: 0		Prediction: 0.0
    Expect: 1		Prediction: 1.0
    Expect: 1		Prediction: 1.0
    Expect: 1		Prediction: 1.0
    Expect: 1		Prediction: 1.0
    Expect: 1		Prediction: 0.0
    Expect: 1		Prediction: 1.0
    Expect: 0		Prediction: 0.0
    Expect: 0		Prediction: 0.0
    Expect: 0		Prediction: 0.0
    Expect: 0		Prediction: 0.0
    Expect: 0		Prediction: 0.0
    Expect: 0		Prediction: 0.0
    Expect: 0		Prediction: 0.0
    Accuracy: 90.0%



![output_19_1](https://user-images.githubusercontent.com/83048295/156874615-212d9bfd-6b33-4084-988a-07a1a6205fa4.png)



### Test on 3D Data Set


```python
print("Running test on 3D data...")
test(test_x3d, train_x3d, test_y3d, train_y3d)
```

    Running test on 3D data...
    Expect: 3		Prediction: 3.0
    Expect: 3		Prediction: 3.0
    Expect: 1		Prediction: 1.0
    Expect: 2		Prediction: 2.0
    Expect: 2		Prediction: 2.0
    Expect: 3		Prediction: 3.0
    Expect: 0		Prediction: 0.0
    Expect: 0		Prediction: 1.0
    Expect: 1		Prediction: 2.0
    Expect: 2		Prediction: 2.0
    Expect: 1		Prediction: 1.0
    Expect: 2		Prediction: 1.0
    Expect: 3		Prediction: 3.0
    Expect: 0		Prediction: 0.0
    Expect: 2		Prediction: 2.0
    Expect: 0		Prediction: 1.0
    Expect: 3		Prediction: 3.0
    Expect: 0		Prediction: 0.0
    Expect: 0		Prediction: 0.0
    Expect: 0		Prediction: 0.0
    Expect: 0		Prediction: 0.0
    Expect: 1		Prediction: 0.0
    Expect: 1		Prediction: 0.0
    Expect: 1		Prediction: 1.0
    Expect: 3		Prediction: 3.0
    Expect: 3		Prediction: 3.0
    Expect: 0		Prediction: 0.0
    Expect: 0		Prediction: 0.0
    Expect: 3		Prediction: 3.0
    Expect: 0		Prediction: 0.0
    Expect: 3		Prediction: 3.0
    Expect: 2		Prediction: 2.0
    Expect: 2		Prediction: 2.0
    Expect: 3		Prediction: 3.0
    Expect: 1		Prediction: 1.0
    Expect: 2		Prediction: 2.0
    Expect: 3		Prediction: 0.0
    Expect: 1		Prediction: 1.0
    Expect: 3		Prediction: 3.0
    Expect: 1		Prediction: 1.0
    Expect: 0		Prediction: 0.0
    Expect: 3		Prediction: 0.0
    Expect: 2		Prediction: 2.0
    Expect: 2		Prediction: 2.0
    Expect: 1		Prediction: 1.0
    Expect: 3		Prediction: 3.0
    Expect: 3		Prediction: 3.0
    Expect: 0		Prediction: 0.0
    Expect: 1		Prediction: 1.0
    Expect: 2		Prediction: 2.0
    Expect: 1		Prediction: 1.0
    Expect: 1		Prediction: 1.0
    Expect: 1		Prediction: 1.0
    Expect: 1		Prediction: 0.0
    Expect: 3		Prediction: 3.0
    Expect: 2		Prediction: 2.0
    Expect: 1		Prediction: 1.0
    Expect: 1		Prediction: 1.0
    Expect: 3		Prediction: 0.0
    Expect: 1		Prediction: 2.0
    Expect: 2		Prediction: 2.0
    Expect: 2		Prediction: 2.0
    Expect: 0		Prediction: 0.0
    Expect: 0		Prediction: 0.0
    Expect: 1		Prediction: 1.0
    Expect: 0		Prediction: 3.0
    Expect: 1		Prediction: 1.0
    Expect: 2		Prediction: 2.0
    Expect: 1		Prediction: 0.0
    Expect: 2		Prediction: 2.0
    Expect: 2		Prediction: 2.0
    Expect: 1		Prediction: 1.0
    Expect: 2		Prediction: 2.0
    Expect: 0		Prediction: 0.0
    Expect: 3		Prediction: 3.0
    Expect: 1		Prediction: 1.0
    Expect: 1		Prediction: 1.0
    Expect: 3		Prediction: 3.0
    Expect: 2		Prediction: 2.0
    Expect: 2		Prediction: 2.0
    Expect: 1		Prediction: 1.0
    Expect: 1		Prediction: 1.0
    Expect: 1		Prediction: 1.0
    Expect: 3		Prediction: 3.0
    Expect: 3		Prediction: 3.0
    Expect: 2		Prediction: 2.0
    Expect: 0		Prediction: 0.0
    Expect: 2		Prediction: 2.0
    Expect: 1		Prediction: 0.0
    Expect: 3		Prediction: 3.0
    Expect: 1		Prediction: 1.0
    Expect: 1		Prediction: 0.0
    Expect: 3		Prediction: 3.0
    Expect: 1		Prediction: 0.0
    Expect: 2		Prediction: 2.0
    Expect: 1		Prediction: 2.0
    Expect: 2		Prediction: 2.0
    Expect: 0		Prediction: 1.0
    Expect: 1		Prediction: 1.0
    Expect: 2		Prediction: 1.0
    Expect: 3		Prediction: 3.0
    Expect: 3		Prediction: 1.0
    Expect: 0		Prediction: 0.0
    Expect: 3		Prediction: 1.0
    Expect: 1		Prediction: 1.0
    Expect: 2		Prediction: 2.0
    Expect: 2		Prediction: 2.0
    Expect: 3		Prediction: 3.0
    Expect: 3		Prediction: 0.0
    Expect: 0		Prediction: 0.0
    Expect: 0		Prediction: 0.0
    Expect: 2		Prediction: 2.0
    Expect: 1		Prediction: 0.0
    Expect: 2		Prediction: 2.0
    Expect: 1		Prediction: 0.0
    Expect: 1		Prediction: 1.0
    Expect: 2		Prediction: 2.0
    Expect: 1		Prediction: 1.0
    Expect: 1		Prediction: 0.0
    Expect: 2		Prediction: 2.0
    Expect: 3		Prediction: 3.0
    Expect: 3		Prediction: 3.0
    Expect: 0		Prediction: 0.0
    Expect: 3		Prediction: 3.0
    Expect: 0		Prediction: 0.0
    Expect: 3		Prediction: 3.0
    Expect: 1		Prediction: 1.0
    Expect: 1		Prediction: 1.0
    Expect: 3		Prediction: 3.0
    Expect: 2		Prediction: 2.0
    Expect: 1		Prediction: 1.0
    Expect: 2		Prediction: 2.0
    Expect: 2		Prediction: 2.0
    Expect: 2		Prediction: 2.0
    Expect: 3		Prediction: 3.0
    Expect: 2		Prediction: 2.0
    Expect: 1		Prediction: 1.0
    Expect: 0		Prediction: 3.0
    Expect: 2		Prediction: 2.0
    Expect: 3		Prediction: 3.0
    Expect: 0		Prediction: 0.0
    Expect: 2		Prediction: 2.0
    Expect: 2		Prediction: 2.0
    Expect: 0		Prediction: 0.0
    Expect: 1		Prediction: 0.0
    Expect: 1		Prediction: 2.0
    Expect: 1		Prediction: 1.0
    Expect: 1		Prediction: 1.0
    Expect: 0		Prediction: 0.0
    Expect: 0		Prediction: 1.0
    Expect: 1		Prediction: 1.0
    Expect: 0		Prediction: 0.0
    Expect: 1		Prediction: 0.0
    Expect: 3		Prediction: 3.0
    Expect: 3		Prediction: 3.0
    Expect: 3		Prediction: 3.0
    Expect: 1		Prediction: 1.0
    Expect: 1		Prediction: 0.0
    Expect: 3		Prediction: 3.0
    Expect: 1		Prediction: 1.0
    Expect: 2		Prediction: 1.0
    Expect: 3		Prediction: 3.0
    Expect: 2		Prediction: 2.0
    Expect: 3		Prediction: 3.0
    Expect: 3		Prediction: 0.0
    Expect: 1		Prediction: 1.0
    Expect: 1		Prediction: 2.0
    Expect: 3		Prediction: 3.0
    Expect: 1		Prediction: 1.0
    Expect: 1		Prediction: 1.0
    Expect: 3		Prediction: 3.0
    Expect: 3		Prediction: 3.0
    Expect: 0		Prediction: 0.0
    Expect: 3		Prediction: 3.0
    Expect: 1		Prediction: 1.0
    Expect: 3		Prediction: 3.0
    Expect: 0		Prediction: 0.0
    Expect: 2		Prediction: 2.0
    Expect: 3		Prediction: 3.0
    Expect: 0		Prediction: 0.0
    Expect: 3		Prediction: 3.0
    Expect: 3		Prediction: 3.0
    Expect: 1		Prediction: 0.0
    Expect: 1		Prediction: 1.0
    Expect: 3		Prediction: 0.0
    Expect: 3		Prediction: 3.0
    Expect: 2		Prediction: 2.0
    Expect: 1		Prediction: 1.0
    Expect: 3		Prediction: 3.0
    Expect: 0		Prediction: 0.0
    Expect: 3		Prediction: 3.0
    Expect: 3		Prediction: 0.0
    Expect: 1		Prediction: 1.0
    Expect: 0		Prediction: 0.0
    Expect: 3		Prediction: 3.0
    Expect: 2		Prediction: 2.0
    Expect: 2		Prediction: 2.0
    Expect: 0		Prediction: 0.0
    Expect: 2		Prediction: 2.0
    Expect: 0		Prediction: 3.0
    Accuracy: 81.0%




![output_21_1](https://user-images.githubusercontent.com/83048295/156874592-f1c91175-f897-4aa9-ab19-634c703571ab.png)
