#一.记录环境配置和调试代码中遇到的困难，以及解决方案
###1.下载mnist文件
#####从库里导入
TensorFlow库和sklearn库里都有mnist文件
比如
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
可以直接用sklearn直接导入文件
#####先下载到本地再导入（我用的是这种,在网上找到了下载的代码）
[![QQ-20231023172154.png](https://i.postimg.cc/28LQtxLf/QQ-20231023172154.png)](https://postimg.cc/Xp3BGdg2)
###2.将image和label合并
神经网络使用Dataloader来加载数据，要求image和label在同一个文件里，但我下载到电脑上的数据是分开的，需要将image和label合并起来。
找到了一个方法：
[![QQ-20231023135044.png](https://i.postimg.cc/0jxSG8tW/QQ-20231023135044.png)](https://postimg.cc/zbt381YK
###3.其他问题
#####标准化和归一化的问题
在加载文件时，我发现网上给的代码有一个/255.0，查了一下是用来归一化的，归一化可以帮助模型更好的捕捉数据，对于一些特征值不同的数据还有缩放作用等等作用。mnist数据集里的归一化就是把从0到255的像素归一到从0到1，所以是除以255.0。
另外，我考虑是不是用标准化更合适，因为在网课里看到的就是标准化。我尝试着在定义customdataset的时候加入image的标准化，后来又在很多地方进行了一些尝试，最后都以失败告终。我在网上找到了可以用标准化的代码，但从加载数据开始就跟我自己的代码相差太大，我就偷懒放弃了。
_ _ _
#二.对初步要求中问题的回答
###1.开放性问题
常见的图片格式有PDF，GIF，JPG等等，网上搜到了BMP，TIFF，SVG等没怎么听过的。
对于存储形式我不是太理解，但mnist的图像是以二进制形式存储的
常用的图像处理编程语言有python，MATLAB，OpenCV等等，其中我们用的python主要依赖于OpenCV、Pillow、scikit-image和NumPy等等的包2来实现图像处理。
OpenCV是一个流行的库，用于计算机视觉和图像处理。（很不幸我好像没用到它T.T）
###2.记录训练结果
#####未归一化和进行归一化的训练准确率确实有差距（大约5%左右）
#####首先是未经过标准化的结果，94%左右
[![QQ-20231023171315.png](https://i.postimg.cc/W3njmKP6/QQ-20231023171315.png)](https://postimg.cc/vgcpb0Tc)
#####然后是经过归一化的结果，竟然达到了99%
[![QQ-20231023180830.png](https://i.postimg.cc/MH2p648y/QQ-20231023180830.png)](https://postimg.cc/xJ52g68C)
#####我们用了两种计算准确率的方法
一种是像iris数据集中给出的那种，计算预测正确的数据占总数据的比例；另一种是task1plus给出的classificationreport函数计算准确率
###3.回答问题
源代码中是用gzip以二进制只读的形式读取数据的，读取之后转化为4维张量，第0维自动计算为样本总数，在后面用Dataloader后每个batch的第一维会转换成batchsize，第1维是通道数，mnist只有一个通道，第2维是高度，第3维是宽度，高度和宽度都是28
使用卷积神经网络进行监督学习
_ _ _

#三.代码和注释都在py文件里了，请大佬过目