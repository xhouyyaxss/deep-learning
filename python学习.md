[TOC] 

# Python

## 安装

- 可以直接安装

- 通过anaconda安装，同时里面自带jupyter等

  - conda:
    - conda  list 列举已经安装的包
    - conda env list  列举已经创建的环境
    - conda ativate envname   激活指定的环境
    - conda deactivate 推出当前环境
    - conda info 显示conda的配置信息，包括创建环境的文件架，镜像地址等
    - conda install package  /   pip  install  package
    - python3.7  tensorflow2.1.0 /2.4.0  pytorch  1.8.1      3.9->2.9.0 
  - https://www.cgohlke.com/   可以在该网站自己下载pip安装的资源

  - 可以修改一开始jupyter打开的文件夹地址
  - 建议修改anaconda的env地址和pkg地址

## 操作

### 基础操作

- a**b a的b次方
- 索引：a[1:-1]: 从左面开始下标为1（左面从0开始）开始到从右面数第1个（右面从-1开始） 区间左闭右开
  - a[0:]: 整个长度
  - a[i:j\:c]: i和j之间（左闭右开）每隔c个取出来
- type(a)判断类型 
- len(a) 求a的长度

### 字符操作

- a.isalpha(): 判断是否是英文字母
- a.isspace():判断是否是空格
- a.isdigit():判断是否是数字

### 字符串操作

- str(a):强制转成字符串
- a*3: 三个相同的字符串连接
- len(a): 求字符串长度
- a.split('b')将a以b分割
- a.join(b) 将a中的数据以a为分割标志进行拼接
- a.replace(c,d)：将a中的c替换成d
- a.upper():将a中的字母全部转化成大写
- a.lower():将a中的字母全部转化成小写
- a.strip()：将字符串中的前后空格全部去掉
- a.lstrip(): 将字符串中的左面的空格全部去掉
- a.rstrip(): 将字符串中的右面的空格全部去掉
- '{}{}{}'.format(a,b,c)格式化输出
- '%s %d %f' % (a,b,c)

### 列表（List）

- a=[]定义List
- a[index]取List数据
- a=list(['1','2','3']) 创建List
- a+b 求并集
- a*b  b个a的list
- del a[0]     del a[3:]   删除指定下标的元素
- b in a  :判断b是否在a中
- a.count('b')  求b在a中有多少个
- a.index('b')  求b在a中的下标
- a.append('b')  添加b
- a.insert(k,'b')  在k出添加b
- a.remove('b')   删除b
- a.pop(index)   可以按照下标删除a中的元素
- a.sort()  排序   sorted(a)
- a.reverse()  将list翻转

### 集合（Set）: 无序的（不按照输入的顺序），会保留下唯一的那些元素

- set(a): 将a转成set
- a=set([a,b,c])
- a.union(b)    a|b   : 求a和b的并集
- a.intersection(b)   a&b:  求a和b的交集
- a.difference(b)    a-b  ： a去掉a和b的公共部分
- a.issubset(b)    a<=b : a是否是b的一部分
- a.add(c) : 添加c元素
- a.update(b) :  将a和b合并
- a.remove(element) : 根据元素删除元素
- a.pop() :  删除最左边的元素

### 字典（Dict）: 本质是由一组键值对构成的一个序列

- a={}   a=dict()  定义一个字典

- a['b']=c   
- a=dict([('c',d),('i',j)])
- a.get(key)  通过key得到value
-  a.get(key,errorMessage)  : 如果没有自己要查找的key可以指定返回信息
- a.pop(key)  可以通过key删除   del a[key]
- 'key'  in a  判断a中有没有key
- a.keys()  ： 返回所有的key值
- a.values() :  返回所有value
- a.items() :  返回所有键值对

### 文件操作

- %%writefile  huo.txt    :  生成一个文件

- import os      print(os.path.abspath)  输出项目的绝对路径

- with open('filename','w/r',encoding=xxxx)  as f  :    该种方法不需要调用close()  并且这时候带上编码之后就能添加中文

  - f.write('xxxxxx')

  - 使用open()需要调用close()方法

- txt.read()  ：将所有内容一下子读出来

- txt.readlines()  :  按行将内容读出来，每行后面跟一个'\n'

### 类

```python
class a:
    number=10
    def __init__(self):  #构造函数
        xxxxxx
    def setAttr(self,attr):
        self.xxxAttr=attr
    def getAttr(self,attr):
        print(self.attr)
    def newM(self): #子类能够重写方法
        xxxxxx
a1=a()
a.xxAttr=xx
del a.xxAttr  #删除某属性
setattr(a1,attr,xx)#设置属性
getattr(a1,attr) #取属性
hasattr(a1,xxAttr) #判断是否有属性
class b(a):  #b继承a   
    xxxxxx
    xxx
    def newM(self):  #重写父类的方法
        xxxxxx
```

### 时间

```python
#时间
import time 
print(time.time())
print(time.localtime(time.time()))#现在的时间
print(time.asctime(time.localtime(time.time())))
#按照自定义格式输出日期
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
```

```python
#日历
import calendar 
#输出指定月份的日历
print(calendar.month(2024,5))
print(help(calendar.week))

#匿名函数
S=lambda a,b:a*b
S(2,3)
```

### numpy

```python
#数组创建 数组中只能存储相同类型的数据  list中可以存储不同类型的数据
a = np.array([[1,2],[3,4],[5,6]])#创建3行2列二维数组。
print(a)
array([[1, 2],
       [3, 4],
       [5, 6]])
a = np.zeros(6)#创建长度为6的，元素都是0一维数组
a = np.zeros((2,3))#创建3行2列，元素都是0的二维数组
a = np.ones((2,3))#创建3行2列，元素都是1的二维数组
a = np.empty((2,3)) #创建3行2列,未初始化的二维数组
a = np.arange(6)#创建长度为6的，元素都是0一维数组array([0, 1, 2, 3, 4, 5])
a = np.arange(1,7,1)#结果与np.arange(6)一样。第一，二个参数意思是数值从1〜6,不包括7.第三个参数表步长为1.
a=np.linspace(0,10,7) # 生成首位是0，末位是10，含7个数的等差数列[  0.           1.66666667   3.33333333   5.         6.66666667  8.33333333  10.        ]
a=np.logspace(0,4,5)#用于生成首位是10**0，末位是10**4，含5个数的等比数列。[  1.00000000e+00   1.00000000e+01   1.00000000e+02   1.00000000e+03 1.00000000e+04]


#数组合并
a = np.array([[1,2],[3,4],[5,6]])
b = np.array([[10,20],[30,40],[50,60]])
np.vstack((a,b))
array([[ 1,  2],
       [ 3,  4],
       [ 5,  6],
       [10, 20],
       [30, 40],
       [50, 60]])
np.hstack((a,b))
array([[ 1,  2, 10, 20],
       [ 3,  4, 30, 40],
       [ 5,  6, 50, 60]])

#不同维数组相加
a = np.array([[1],[2]])
print(a)
array([[1],
       [2]])
	   
b=([[10,20,30]])#生成一个list，注意，不是np.array。
print(b)
[[10, 20, 30]]

print(a+b)
array([[11, 21, 31],
       [12, 22, 32]])
	   
c = np.array([10,20,30])
print(c)
array([10, 20, 30])

print(c.shape)
(3,)

print(a+c)
array([[11, 21, 31],
       [12, 22, 32]])

#删除行列
a = np.array([[1,2],[3,4],[5,6]])
np.delete(a,1,axis = 0)#删除a的第二行。
array([[1, 2],
       [5, 6]])
	   
np.delete(a,(1,2),0)#删除a的第二，三行。
array([[1, 2]])

np.delete(a,1,axis = 1)#删除a的第二列。  axis的作用表示按哪一维进行操作
array([[1], 
       [3],
       [5]])

#argmax返回的是最大数的索引
import numpy as np
a = np.array([3, 1, 2, 4, 6, 1])
print(np.argmax(a)) #4

import numpy as np
a = np.array([[1, 5, 5, 2],
              [9, 6, 2, 8],
              [3, 7, 9, 1]])
print(np.argmax(a, axis=0))  #(1,2,2,1)
print(np.argmax(a, axis=1))  #(1,0,2)

#强制类型转换
a.astype(float)

#列表中的切片操作
a=list([[1,2],[3,4]])
a[:,:-1]
#np.array中可以a[:,index]对某一列进行选取
可以对每一维进行切片
```

