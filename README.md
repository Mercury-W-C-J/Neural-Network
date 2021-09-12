## 前置库的导入

我们使用的库为以下一些基本库

```python
from tqdm import trange
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import math
from numpy import matlib
```

## 数据导入与预处理

需要将数据进行导入->去除不要的变量->哑变量转换->数据标准化

## 神经网络

我们采用的是面向对象编程，接下来我们将简单介绍我们的类 NeuralNetwork

```python
class NeuralNetwork(activation=None)
```

###### Parameters:

- activation=**'sigmoid'** or **'relu'** or **'tanh'**

  决定最后输出层的激活函数

###### Methods：

- ```python
  NeuralNetwork.add_hidden_layer(nodes=None,activation_function=None)
  ```

  - nodes：**节点数量**
  - activation_function=**'sigmoid'** or **'relu'** or **'tanh'**

  对模型添加一层隐藏层，输入节点数以及激活函数

- ```python
  NeuralNetwork.train(x_train=None,y_train=None,LR=None,iteration=None)
  ```

  - x_train,y_train ：**训练集**
  - LR ：**学习率**
  - iteration：**循环次数**

  根据训练集、学习率以及训练次数对神经网络模型进行训练。 

- ```python
  NeuralNetwork.predict(x_test=None,y_test=None)
  ```

  - x_test,y_test ：**测试集**

  输入测试集进行预测，预测结果储存于**NeuralNetwork.result**

- ```python
  NeuralNetwork.score(threshold=None,y=None)
  ```

  - threshold：**预测的阈值**
  - y：**预测的真实值**

  根据预测的结果，进行对各项指标的评估（Accuracy，Precision，Recall，TNR），返回为一个DataFrame。

### 示例

- ```python
  a=NeuralNetwork('sigmoid')
  a.add_hidden_layer(3, 'relu')
  a.add_hidden_layer(4,'sigmoid')
  a.train(X_train,y_train,0.01,10)
  a.predict(X_test,y_test)
  hahah=a.score(0.5, y_test)
  result=a.result
  ```

### Early stopping

###### Methods:

- ```python
  NeuralNetwork.train_early_stop(x_train=None,y_train=None,LR=None)
  ```

  - x_train,y_train: **训练集**
  - LR: **学习率**

### 示例

- ```python
  a=NeuralNetwork('sigmoid')
  a.add_hidden_layer(3, 'relu')
  a.add_hidden_layer(4,'sigmoid')
  a.train_early_stop(X_train,y_train,0.01)
  a.predict(X_test,y_test)
  hahah=a.score(0.5, y_test)
  result=a.result
  ```

## 随机神经网络

我们采用新的类randomforest来使用随机神经网络

```python
class randomforest()
```

###### Methods:

- ```python
  randomforest.random(X,y,X_test,y_test,feature,number,node):
  ```

  - X,y: **训练集**
  - X_test,y_test: **测试集**
  - feature: **给予每个子模型的随机feature 个数**
  - number： **子模型数量**
  - node: **子模型隐藏层的节点数**

  根据输入的子模型的特征数、子模型个数以及节点数，在训练集上进行训练模型，并且在测试集上进行预测。

- ```python
  randomforest.randomscore(threshold,y):
  ```

  - threshold: **预测的阈值**
  - y: **预测的真实值**

  根据预测的结果，进行对各项指标的评估（Accuracy，Precision，Recall，TNR），返回为一个DataFrame。

### 示例

- ```python
  l=randomforest()
  l.random(X_train, y_train, X_test_data, y_test_data, 5, 10,3)
  hahah_random=l.randomscore(0.5,y_test_data)
  result=l.predict_sum
  ```

