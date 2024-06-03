[TOC]

# PyTorch

### 安装

- python3.7  tensorflow2.1.0 /2.4.0  pytorch  1.8.0      3.9->2.9.0 
- https://pytorch.org/get-started/locally/  在官网选择对应的版本并且选择CPU或者GPU版本的

## Q

- anaconda创建的环境中打开jupyter  如果pyzmq版本过高生成文件会闪退 可以使用conda unstall pyzmq    pip install pyzmq==19.0.2
- 在自己创建的环境中启动jupyter 可以使用 jupyter notebook --notebook- dir='d:\\python\\jupyter'来将jupyter的启动地址设置成想要的（ipython 7.31.1 jupyter_client 7.4.9 jupyrer_server 1.23.1）   2.x版本jupyter用serverApp root dir
- ![image-20240531013509834](D:\postgraduate\deeplearing\images\matplotlib和python版本)

## 使用

- 导入   import torch 
- torch,rand(5,3) 一个5*3的随机矩阵  返回一个张量(Tensor) 包含了区间[0,1)中均匀分布抽取的一组数  
- torch.randn(*size,out=None)  返回一个size大小的包含了从标准正态分布（均值0，方差1,高斯白噪声）中抽取的一组随机数
- torch.zeros(5,3,dtype=long)  全零的矩阵
- torch.tensor([5.3,3]) 直接传入数据
- x.size()展示矩阵大小
- x+y 矩阵相加    torch.add(x,y)
- x=torch.randn(4,4)    x.view(16)   x.view(-1,8)   -1代表自己运算  16/8=2  因此大小为torch.Size([2,8])
- 和numpy协同操作
  - a=torch.ones(5)   b=a.numpy()     res:array([1., 1., 1., 1., 1.], dtype=float32)
  - a=numpy.ones(5)   b=torch.from_numpy(a)    res:tensor([1., 1., 1., 1., 1.], dtype=torch.float64)

- x=torch.randn(3,4,requires_grad=True（表示可以对x进行求导）)
- torch.sum(input, dim, keepdim=False, *, dtype=None) → Tensor   
  - input：输入的张量
  - dim：求和的维度，可以是一个列表,也就是可以同时接收多个维度，并可同时在这些维度上进行指定操作。(dim表示对哪一个维度进行求和   eg:torch.sum(x,y,z)  x块y行每行z列  dim=0对x  dim=1对y  dim=2对z)    
  - keepdim：默认为False，若keepdim=True，则返回的Tensor除dim之外的维度与input相同。因为求和之后这个dim的元素个数为１，所以要被去掉，如果要保留这个维度，则应当keepdim=True。
- sum.backword(retain_graph=True)#不清空会将每次算的梯度累加起来 ，sum需要是标量  
- torch.grad  求梯度

## Demo

- 先定义模型：

  ```python
  #简便方法
  class LinearRegressionModel(nn.Module):
      def __init__(self,input_dim,output_dim):
          #调用父类nn中的方法
          super(LinearRegressionModel,self).__init__()
          #本demo中直接调用全连接层
          self.linear = nn.Linear(input_dim,output_dim)
      #前向传播函数  其中可以自己定义模型的处理顺序，该例子中只有全连接层
      def forward(self,x):
          out=self.linear(x)
          return out
  #输入的维度 本例中输入是x_values中的一个数，为一维，输出是wx+b也是一维
  input_dim=1
  #输出的维度
  output_dim=1
  #创建模型
  model=LinearRegressionModel(input_dim,output_dim)
  
  #设置参数以及损失函数
  epochs=1000   #全部训练数据每训练一次为一代
  learning_rate=0.01 #学习率用来控制梯度下降的幅度
  #优化器
  optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
  criterion=nn.MSELoss()
  
  #进行训练
  for epoch in range(epochs):
      #迭代次数计数器
      # epoch+=1
      #转化成tensor才能进行训练
      inputs=torch.from_numpy(x_train)
      labels=torch.from_numpy(y_train)
      
      #梯度每一代要清零
      optimizer.zero_grad()
      #前向传播
      output=model.forward(inputs)
      #计算损失
      loss=criterion(output,labels)
      #反向传播
      loss.backward()
      #更新权重参数
      optimizer.step()
      
      #每迭代50次打印一下
      if epoch%50==0:
          print('Epoch:',epoch,'Loss:',loss.item())
      
  ```

- 指定参数和损失函数

  - SGD：随机梯度下降（**Stochastic Gradient Descent**）**是一种用于训练机器学习算法的优化算法**，最值得注意的是深度学习中使用的人工神经网络。**该算法的工作是找到一组内部模型参数**，这些参数在某些性能测量中表现良好，例如对数损失或均方误差。
  - **优化是一种搜索过程，您可以将此搜索视为学习**。优化算法称为“ 梯度下降 ”，**其中“ 梯度 ”是指误差梯度或误差斜率的计算**，“下降”是指沿着该斜率向下移动到某个最小误差水平。该算法是迭代的。这意味着搜索过程发生在多个不连续的步骤上，每个步骤都希望略微改进模型参数。
  - 每一步都需要使用模型和当前的一组内部参数对一些样本进行预测，将预测与实际预期结果进行比较，**计算误差，并使用误差更新内部模型参数**。该**更新过程**对于不同的算法是不同的，但是在**人工神经网络的情况下，使用反向传播更新算法。**
  - **Sample**是单行数据。它包含<mark>输入到算法中的输入和用于与预测进行比较并计算错误的输出</mark>。训练数据集由许多行数据组成，例如许多Sample。Sample也可以称为实例，观察，输入向量或特征向量。
  - Batch大小是一个超参数，用于定义在更新内部模型参数之前要处理的样本数。将批处理视为循环迭代一个或多个样本并进行预测。在批处理结束时，将预测与预期输出变量进行比较，并计算误差。从该错误中，更新算法用于改进模型，例如沿误差梯度向下移动。
    训练数据集可以分为一个或多个Batch。当所有训练样本用于创建一个Batch时，学习算法称为批量梯度下降。当批量是一个样本的大小时，学习算法称为随机梯度下降。当批量大小超过一个样本且小于训练数据集的大小时，学习算法称为小批量梯度下降。batch大小相当于完成batch大小的训练进行参数更新
    - 批量梯度下降。批量大小=训练集的大小（总样本数量）
    - 随机梯度下降。批量大小= 1（每次一个样本）
    - 小批量梯度下降。1 <批量大小<训练集的 大小
      在小批量梯度下降的情况下，流行的批量大小包括32,64和128个样本。可能会在文献和教程中看到这些值在模型中使用。
  - （1）batchsize：批大小。在深度学习中，一般采用SGD训练，即每次训练在训练集中取batchsize个样本训练；
    （2）iteration：1个iteration等于使用batchsize个样本训练一次；
    （3）epoch：1个epoch等于使用训练集中的全部样本训练一次；
    举个例子，训练集有1000个样本，batchsize=10，那么：
    训练一个epoch是100个batch，也就是这1个epoch中iteration是100。

## Tensor常见的形式

- dim=0:scalar  标量，一个数值  常用于Loss

- dim=1:vector  向量   常用于偏置和线性层输入

- dim=2matrix  矩阵 常用于带有batch的线性层输入

- dim=3 常用于RNN的输入，假设输入一句话有10个单词，[哪个单词，哪句话，单词编码]

- dim=4常用于CNN中

- :n-dimensional tensor n维度张量

- tensor的数据类型

  | python |   PyTorch   |
  | :----: | :---------: |
  |  int   |  intTensor  |
  | float  | floatTensor |

  pytorch中没有对string的支持，使用one-hot或者Embedding来表示

$$
\vec{a}
$$

### Scalar

```python
import torch
from torch import tensor

#scalar  括号中有几个中括号就表示dim等于几
x=tensor(65.)
#获取形状
y=x.shape
print(y)
type(y)
print(type(y))
#打印维度
x.dim()
x*=2
#将去重tensor类型中的数值
x.item()

```

### Vector

```python
v=tensor([1.5,-0.,3.0])
n=tensor([[[]]]) #
n.size() #torch.Size([1, 1, 0])  里面的数字表示每一维中数据的个数
# v.size()
# v.dim()
```

### Matrix

```python
M=tensor([[1.,2.],[3.,4.]])
M.size() #torch.Size([2, 2])
M.shape #torch.Size([2, 2])
# M.dim()

M.matmul(M) #与自身的矩阵相乘
M*M #求矩阵内积：两个矩阵对应分量乘积之和
```



### hub模块

**GITHUB:https://github.com/pytorch/hub**

**模型：https://pytorch.org/hub/research-models**

能够使用别人训练好的模型

