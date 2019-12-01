# 问题九十一至一百

## 问题九十一：$k-$平均聚类算法进行减色处理第一步----按颜色距离分类

对`imori.jpg`利用 $k-$平均聚类算法进行减色处理。

在问题六中涉及到了减色处理，但是在问题六中事先确定了要减少的颜色。这里，$k-$平均聚类算法用于动态确定要减少的颜色。

算法如下：

1. 从图像中随机选取$K$个$\text{RGB}$分量（这我们称作类别）。

2. 将图像中的像素分别分到颜色距离最短的那个类别的索引中去，色彩距离按照下面的方法计算：
   $$
   \text{dis}=\sqrt{(R-R')^2+(G-G')^2+(B-B')^2}
   $$

3. 计算各个索引下像素的颜色的平均值，这个平均值成为新的类别；
4. 如果原来的类别和新的类别完全一样的话，算法结束。如果不一样的话，重复步骤2和步骤3；
5. 将原图像的各个像素分配到色彩距离最小的那个类别中去。

完成步骤1和步骤2。
- 类别数$K=5$；
- 使用`reshape((HW, 3))`来改变图像大小之后图像将更容易处理；
- 步骤1中，对于`np.random.seed(0)`，使用`np.random.choice(np.arrange(图像的HW), 5, replace=False)`；
- 现在先不考虑步骤3到步骤5的循环。


```bash
# 最初选择的颜色
[[140. 121. 148.]
 [135. 109. 122.]
 [211. 189. 213.]
 [135.  86.  84.]
 [118.  99.  96.]]
```

最初に選ばれた色との色の距離でクラスのインデックスをつけたもの(アルゴリズム2)。
解答では0-4にインデックスの値をx50にして見やすいようにしている。

| 输入 (imori.jpg) | 输出(answers/answer_91.jpg) |
| :--------------: | :-------------------------: |
|  ![](imori.jpg)  | ![](answers/answer_91.jpg)  |

答案 >> [answers/answer_91.py](answers/answer_91.py)

## 问题九十二：利用 $k-$平均聚类算法进行减色处理第二步----减色处理

实现算法的第3到5步。

```bash
# 选择的颜色
[[182.86730957 156.13246155 180.24510193]
 [156.75152588 123.88993835 137.39085388]
 [227.31060791 199.93135071 209.36465454]
 [ 91.9105835   57.94448471  58.26378632]
 [121.8759613   88.4736557   96.99688721]]
```

减色处理可以将图像处理成手绘风格。如果$k=10$，则可以在保持一些颜色的同时将图片处理成手绘风格。

现在，$k=5$的情况下试着将`madara.jpg`进行减色处理。

| 输入 (imori.jpg) | 输出(answers/answer_92.jpg) | k=10(answers/answer_92_k10.jpg) | 输入2 (madara.jpg) | 输出(answers/answer_92_m.jpg) |
| :--------------: | :-------------------------: | :-----------------------------: | :----------------: | :---------------------------: |
|  ![](imori.jpg)  | ![](answers/answer_92.jpg)  | ![](answers/answer_92_k10.jpg)  |  ![](madara.jpg)   | ![](answers/answer_92_m.jpg)  |

答案 >> [answers/answer_92.py](answers/answer_92.py)


## 问题九十三：准备机器学习的训练数据第一步——计算$\text{IoU}$

从这里开始我们准备机器学习用的训练数据。

我的最终目标是创建一个能够判断图像是否是蝾螈的脸的判别器。因此，我们需要蝾螈的脸部图像和非蝾螈脸部的图像。我们需要编写程序来准备这样的图像。

为此，有必要从单个图像中用矩形框出蝾螈头部（即Ground-truth），如果随机切割的矩形与Ground-truth在一定程度上重合，那么这个矩形框处就是蝾螈的头。

重合程度通过检测评价函数$\text{IoU}$（Intersection over Union）来判断。通过下式进行计算：
$$
\text{IoU}=\frac{|\text{Rol}|}{|R_1 + R_2 - \text{Rol}|}
$$
其中：

* $R_1$：Ground-truth的范围；
* $R_2$：随机框出来的矩形的范围；
* $\text{Rol}$：$R_1$和$R_2$重合的范围。

计算以下两个矩形的$\text{IoU}$吧！

```python
# [x1, y1, x2, y2] x1,y1...矩形左上的坐标  x2,y2...矩形右下的坐标
a = np.array((50, 50, 150, 150), dtype=np.float32)
b = np.array((60, 60, 170, 160), dtype=np.float32)
```

答案

```bash
0.627907
```
答案 >> [answers/answer_93.py](answers/answer_93.py)

## 问题九十四：准备机器学习的训练数据第二步——随机裁剪（Random Cropping）

下面，通过从`imori1.jpg`中随机裁剪图像制作训练数据。

这里，从图像中随机切出200个$60\times60$的矩形。

并且，满足下面的条件：

1. 使用`np.random.seed(0)`，求出裁剪的矩形的左上角座标`x1 = np.random.randint(W-60)`和`y1=np.random.randint(H-60)`；
2. 如果和 Ground-truth （`gt = np.array((47, 41, 129, 103), dtype=np.float32)`）的$\text{IoU}$大于$0.5$，那么就打上标注$1$，小于$0.5$就打上标注$0$。

答案中，标注$1$的矩形用红色画出，标注$0$的矩形用蓝色的线画出，Ground-truth用绿色的线画出。我们简单地准备蝾螈头部和不是头部的图像。

| 输入 (imori_1.jpg) | 输出(answers/answer_94.jpg) |
| :----------------: | :-------------------------: |
|  ![](imori_1.jpg)  | ![](answers/answer_94.jpg)  |

答案 >> [answers/answer_94.py](answers/answer_94.py)

## 问题九十五：神经网络（Neural Network）第一步——深度学习（Deep Learning）

将神经网络作为识别器，这就是现在流行的深度学习。

下面的代码是包含输入层、中间层（Unit 数：64）、输出层（1）的网络。这是实现异或逻辑的网络。网络代码参照了[这里](https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6)：

```python
import numpy as np

np.random.seed(0)

class NN:
    def __init__(self, ind=2, w=64, outd=1, lr=0.1):
        self.w1 = np.random.normal(0, 1, [ind, w])
        self.b1 = np.random.normal(0, 1, [w])
        self.wout = np.random.normal(0, 1, [w, outd])
        self.bout = np.random.normal(0, 1, [outd])
        self.lr = lr

    def forward(self, x):
        self.z1 = x
        self.z2 = sigmoid(np.dot(self.z1, self.w1) + self.b1)
        self.out = sigmoid(np.dot(self.z2, self.wout) + self.bout)
        return self.out

    def train(self, x, t):
        # backpropagation output layer
        #En = t * np.log(self.out) + (1-t) * np.log(1-self.out)
        En = (self.out - t) * self.out * (1 - self.out)
        grad_En = En #np.array([En for _ in range(t.shape[0])])
        grad_wout = np.dot(self.z2.T, En)
        grad_bout = np.dot(np.ones([En.shape[0]]), En)
        self.wout -= self.lr * grad_wout#np.expand_dims(grad_wout, axis=-1)
        self.bout -= self.lr * grad_bout

        # backpropagation inter layer
        grad_u1 = np.dot(En, self.wout.T) * self.z2 * (1 - self.z2)
        grad_w1 = np.dot(self.z1.T, grad_u1)
        grad_b1 = np.dot(np.ones([grad_u1.shape[0]]), grad_u1)
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

train_x = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
train_t = np.array([[0], [1], [1], [0]], dtype=np.float32)

nn = NN(ind=train_x.shape[1])

# train
for i in range(1000):
    nn.forward(train_x)
    nn.train(train_x, train_t)

# test
for j in range(4):
    x = train_x[j]
    t = train_t[j]
    print("in:", x, "pred:", nn.forward(x))
```

，我们可以再增加一层中间层进行学习和测试。

答案：

```bash
in: [0. 0.] pred: [0.03724313]
in: [0. 1.] pred: [0.95885516]
in: [1. 0.] pred: [0.9641076]
in: [1. 1.] pred: [0.03937037]
```

答案 >> [answers/answer_95.py](answers/answer_95.py)

## 问题九十六：神经网络（Neural Network）第二步——训练

，将问题九十四中准备的200个训练数据的HOG特征值输入到问题九十五中的神经网络中进行学习。

，对于输出大于 0.5 的打上标注 1，小于 0.5 的打上标注 0，对训练数据计算准确率。训练参数如下：

- $\text{learning rate}=0.01$；
- $\text{epoch}=10000$​；
- 将裁剪的图像调整为$32\times32$，并计算 HOG 特征量（HOG 中1个cell的大小为$8\times8$）。

```bash
Accuracy >> 1.0 (200.0 / 200)
```

答案 >> [answers/answer_96.py](answers/answer_96.py)

## 问题九十七：简单物体检测第一步----滑动窗口（Sliding Window）+HOG

从这里开始进行物体检测吧！

物体检测是检测图像中到底有什么东西的任务。例如，图像在$[x_1, y_1, x_2, y_2]$处有一只狗。像这样把物体圈出来的矩形我们称之为Bounding-box。

下面实现简单物体检测算法：

1. 从图像左上角开始进行滑动窗口扫描；
2. 在滑动的过程中，会依次圈出很多矩形区域；
3. 裁剪出每个矩形区域对应的图像，并对裁剪出的图像提取特征（HOG，SIFT等）；
4. 使用分类器（CNN，SVM等）以确定每个矩形是否包含目标。

这样做的话，会得到一些裁剪过的图像和其对应的矩形的坐标。目前，物体检测主要通过深度学习（Faster R-CNN、YOLO、SSD等）进行，但是这种滑动窗口方法在深度学习开始流行之前已成为主流。为了学习检测的基础知识我们使用滑动窗口来进行检测。

我们实现步骤1至步骤3。

在`imorimany.jpg`上检测蝾螈的头吧！条件如下：

- 矩形使用以下方法表示：
```python
# [h, w]
recs = np.array(((42, 42), (56, 56), (70, 70)), dtype=np.float32)
```
- 滑动步长为4个像素（每次滑动一个像素固然是好的，但这样需要大量计算，处理时间会变长）；
- 如果矩形超过图像边界，改变矩形的形状使其不超过图像的边界；
- 将裁剪出的矩形部分大小调整为$32\times32$；
- 计算HOG特征值时 cell 大小取$8\times8$。

答案 >> [answers/answer_97.py](answers/answer_97.py)

## 问题九十八：简单物体检测第二步——滑动窗口（Sliding Window）+ NN

对于`imorimany.jpg`，将问题九十七中求得的各个矩形的HOG特征值输入问题九十六中训练好的神经网络中进行蝾螈头部识别。

在此，绘制$\text{Score}$（即预测是否是蝾螈头部图像的概率）大于$0.7$的矩形。

下面的答案内容为检测矩形的$[x1, y1, x2, y2, \text{Score}]$：

```bash
[[ 27.           0.          69.          21.           0.74268049]
[ 31.           0.          73.          21.           0.89631011]
[ 52.           0.         108.          36.           0.84373157]
[165.           0.         235.          43.           0.73741703]
[ 55.           0.          97.          33.           0.70987278]
[165.           0.         235.          47.           0.92333214]
[169.           0.         239.          47.           0.84030839]
[ 51.           0.          93.          37.           0.84301022]
[168.           0.         224.          44.           0.79237294]
[165.           0.         235.          51.           0.86038564]
[ 51.           0.          93.          41.           0.85151915]
[ 48.           0.         104.          56.           0.73268318]
[168.           0.         224.          56.           0.86675902]
[ 43.          15.          85.          57.           0.93562483]
[ 13.          37.          83.         107.           0.77192307]
[180.          44.         236.         100.           0.82054873]
[173.          37.         243.         107.           0.8478805 ]
[177.          37.         247.         107.           0.87183443]
[ 24.          68.          80.         124.           0.7279032 ]
[103.          75.         145.         117.           0.73725153]
[104.          68.         160.         124.           0.71314282]
[ 96.          72.         152.         128.           0.86269195]
[100.          72.         156.         128.           0.98826957]
[ 25.          69.          95.         139.           0.73449174]
[100.          76.         156.         132.           0.74963093]
[104.          76.         160.         132.           0.96620193]
[ 75.          91.         117.         133.           0.80533424]
[ 97.          77.         167.         144.           0.7852362 ]
[ 97.          81.         167.         144.           0.70371708]]
```

| 输入 (imori_many.jpg) | 输出(answers/answer_98.jpg) |
| :-------------------: | :-------------------------: |
|  ![](imori_many.jpg)  | ![](answers/answer_98.jpg)  |

解答 >> [answers/answer_98.py](answers/answer_98.py)

## 问题九十九：简单物体检测第三步——非极大值抑制（Non-Maximum Suppression）

虽然使用问题九十七中的方法可以粗略地检测出目标，但是 Bounding-box 的数量过多，这对于后面的处理流程是十分不便的。因此，使用非极大值抑制（Non-Maximum Suppression）减少矩形的数量。

NMS是一种留下高分Bounding-box的方法，算法如下：

1. 将Bounding-box的集合$B$按照$\text{Score}$从高到低排序；
2. $\text{Score}$最高的记为$b_0$；
3. 计算$b_0$和其它Bounding-box的$\text{IoU}$。从$B$中删除高于$\text{IoU}$阈值$t$的Bounding-box。将$b_0$添加到输出集合$R$中，并从$B$中删除。
4. 重复步骤2和步骤3直到$B$中没有任何元素；
5. 输出$R$。

在问题九十八的基础上增加NMS（阈值$t=0.25$），并输出图像。请在答案中Bounding-box的左上角附上$\text{Score}$。

不管准确度如何，这样就完成了图像检测的一系列流程。通过增加神经网络，可以进一步提高检测精度。

| 输入 (imori_many.jpg) | NMS前(answers/answer_98.jpg) | NMS後(answers/answer_99.jpg) |
| :-------------------: | :--------------------------: | :--------------------------: |
|  ![](imori_many.jpg)  |  ![](answers/answer_98.jpg)  |  ![](answers/answer_99.jpg)  |

解答 >> [answers/answer_99.py](answers/answer_99.py)

## 问题一百：简单物体检测第四步——评估（Evaluation）：Precision、Recall、F-Score、mAP

最后是第100个问题！ ！ ！

我们对检测效果作出评估。

検出はBounding-boxとそのクラスの２つが一致していないと、精度の評価ができない。对于检测效果，我们有Recall、Precision、F-Score、mAP等评价指标。

> 下面是相关术语中日英对照表：
>
> |     中文      |  English  | 日本語 |
> | :-----------: | :-------: | :----: |
> |    准确率     | Accuracy  | 正確度 |
> |  精度/查准率  | Precision | 適合率 |
> | 召回率/查全率 |  Recall   | 再現率 |
>
> 我个人认为“查准率&查全率”比“精度&召回率”更准确，所以下面按照“查准率&查全率”翻译。
>
> ——gzr

**查全率（Recall）——在多大程度上检测到了正确的矩形？用来表示涵盖了多少正确答案。取值范围为$[0,1]$：**
$$
\text{Recall}=\frac{G'}{G}
$$
其中：

* $G'$——Ground-truthの中で検出のどれかとIoUが閾値t以上となったGround-truthの数；
* $G$——Ground-truthの矩形の数。

**查准率（Precision）——表示结果在多大程度上是正确的。取值范围为$[0,1]$：**
$$
\text{Precision}=\frac{D'}{D}
$$
其中：

* $D‘$——検出の中で、Ground-truthのどれかとIoUが閾値t以上となった検出の数；
* $D$——検出の数。

**F-Score——是查全率（Recall）和查准率（Precision）的调和平均。可以表示两者的平均，取值范围为$[0,1]$：**
$$
\text{F-Score}=\frac{2\cdot \text{Recall}\cdot \text{Precision}}{\text{Recall}+\text{Precision}}
$$
在文字检测任务中，通常使用Recall、Precision和F-Score等指标进行评估。

**mAP——Mean Average Precision。在物体检测任务中，常常使用mAP进行效果评价。mAP的计算方法稍稍有点复杂：**

1. 各検出に関してGround-truthとの$\text{IoU}$が閾値t以上かどうかを判断して、表を作成する。

   | Detect  |                        judge                         |
   | :-----: | :--------------------------------------------------: |
   | detect1 | 1   (1はGround-truthとの$\text{IoU}$>=tとなったもの) |
   | detect2 | 0   (0はGround-truthとの$\text{IoU}$<tとなったもの)  |
   | detect3 |                          1                           |

2. mAP = 0として、上から順に見て、judgeが1の時は、見ているものの上すべてに関して、Precisionを計算し、mAPに加算する。

3. 上から順に2を行い、全て行ったら、加算回数でmAPを割る。

上面就是求mAP的方法了。对于上面的例子来说：
1. detect1 が1なので、Precisionを求める。Precision = 1/1 = 1なので、mAP = 1
2. detect2 が0なので、無視。
3. detect3 が1なので、Precisionを求める。Precision = 2/3 = 0.67 なので、 mAP = 1 + 0.67 = 1.67
4. mAPに加算したのは計2回なので、mAP = 1.67 / 2 = 0.835
となる。

令阈值$t=0.5$，计算查全率、查准率、F-Score和mAP吧。

设下面的矩形为Ground-truth：

```python
# [x1, y1, x2, y2]
GT = np.array(((27, 48, 95, 110), (101, 75, 171, 138)), dtype=np.float32)
```

请将与Ground-truth的$\text{IoU}$为$0.5$以上的矩形用红线表示，其他的用蓝线表示。

解答
```bash
Recall >> 1.00 (2.0 / 2)
Precision >> 0.25 (2.0 / 8)
F-Score >>  0.4
mAP >> 0.0625
```

| 输入 (imori_many.jpg) | GT(answers/answer_100_gt.jpg)  | 输出(answers/answer_100.jpg) |
| :-------------------: | :----------------------------: | :--------------------------: |
|  ![](imori_many.jpg)  | ![](answers/answer_100_gt.jpg) | ![](answers/answer_100.jpg)  |

解答 >> [answers/answer_100.py](answers/answer_100.py)
