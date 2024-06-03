# Matplotlib

## 基础概念

![../../_images/anatomy.png](D:\postgraduate\deeplearing\images\matplotlib基础概念)

![img](D:\postgraduate\deeplearing\images\常用图形类型)

![img](D:\postgraduate\deeplearing\images\绘图步骤)

## Axex

Axis 指 x、y 坐标轴等（如果有三维那就还有 z 轴），代表的是 “坐标轴”。而 Axes 在英文里是 Axis 的复数形式，也就是说 axes 代表的其实是 figure 当中的一套坐标轴。之所以说一套而不是两个坐标轴，是因为如果你画三维的图，axes 就代表 3 根坐标轴了。所以，在一个 figure 当中，每添加一次 subplot ，其实就是添加了一套坐标轴，也就是添加了一个 axes，放在二维坐标里就是你添加了两根坐标轴，分别是 x 轴和 y 轴。所以当你只画一个图的时候，plt.xxx 与 ax.xxx 其实都是作用在相同的图上的。
