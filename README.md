# 图像处理 100 问！！

> 日本语本当苦手，翻译出错还请在 issue 指正。代码算法方面的问题请往原[ repo ](https://github.com/yoyoyo-yo/Gasyori100knock)提。现阶段我并没有做这些题目（捂脸……），只是翻译而已，因此算法细节可能没有翻译到位。不太好翻译的地方我也会在一定程度上意译~~自行发挥~~，请各位谅解。后续在写代码的途中会对翻译有所更正。
>
> 我会尽量附上英文术语，有翻译不清楚的地方还请参照原文、英语及代码。
>
> 由于使用的编辑器的`MarkDown`标准与`GitHub`的标准有稍许不一样，因此可能出现乱码的情况，建议使用[Typora](https://typora.io/)阅读或阅读 PDF 格式文件。
>
> > 如果公式没有重排一遍，那就代表我没有看懂公式或者没有看懂日语或者既没有看懂公式又没有看懂日语（姚明笑.jpg），请谨慎参考。
>
> 感谢！
>
> ——gzr

英文版本在[这里]( https://github.com/KuKuXia/Image_Processing_100_Questions)，谢谢[KuKuXia](https://github.com/KuKuXia)桑为我做英文翻译。

为图像处理初学者设计的 100 个问题完成了啊啊啊啊啊(´；ω；｀)

和蝾螈一起学习基本的图像处理知识，理解图像处理算法吧！解答这里的提出的问题请不要调用`OpenCV`的`API`，**自己动手实践吧**！虽然包含有答案，但不到最后请不要参考。一边思考，一边完成这些问题吧！

- **问题不是按照难易程度排序的。虽然我尽可能地提出现在流行的问题，但在想不出新问题的情况下也会提出一些没怎么听说过的问题（括弧笑）。**

- **这里的内容参考了各式各样的文献，因此也许会有不对的地方，请注意。**如果发现了错误还请 pull requests ！！

- 【注意】使用这个页面造成的任何事端，本人不负任何责任。

  > 俺也一样。使用这个页面造成的任何事端，本人不负任何责任。
  >
  > ——gzr

请根据自己的喜好，选择 Python 或者 C++ 来进行尝试吧。

> 深度学习无限问请点击[这里](https://github.com/yoyoyo-yo/DeepLearningMugenKnock)。

## Recent
- 2019.3.13 Q95-100 Neural Networkを修正
- 2019.3.8 Questions_01_10 にC++の解答を追加！
- 2019.3.7 TutorialにC++用を追加　そろそろC++用の答えもつくろっかなーと
- 2019.3.5 各Questionの答えをanswersディレクトリに収納
- 2019.3.3 Q.18-22. 一部修正
- 2019.2.26 Q.10. メディアンフィルタの解答を一部修正
- 2019.2.25 Q.9. ガウシアンフィルタの解答を一部修正
- 2019.2.23 Q.6. 減色処理のREADMEを修正
- 2019.1.29 HSVを修正

## 首先

打开终端，输入以下指令。使用这个命令，你可以将整个目录完整地克隆到你的计算机上。

```bash
$ git clone https://github.com/yoyoyo-yo/Gasyori100knock.git
```

然后，选择你喜欢的 Python 或者 C++，阅读下一部分——Tutorial！

## [Tutorial](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Tutorial)

|      |      内容      |                            Python                            |                             C++                              |
| :--: | :------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  1   |      安装      | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Tutorial) | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Tutorial/README_opencv_c_install.md) |
|  2   | 读取、显示图像 | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Tutorial#%E7%94%BB%E5%83%8F%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%81%BF%E8%A1%A8%E7%A4%BA) | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Tutorial/README_opencv_c_install.md#%E7%94%BB%E5%83%8F%E8%AA%AD%E3%81%BF%E8%BE%BC%E3%81%BF) |
|  3   |    操作像素    | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Tutorial#%E7%94%BB%E7%B4%A0%E3%82%92%E3%81%84%E3%81%98%E3%82%8B) | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Tutorial/README_opencv_c_install.md#%E7%94%BB%E7%B4%A0%E3%82%92%E3%81%84%E3%81%98%E3%82%8B) |
|  4   |    拷贝图像    | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Tutorial#%E7%94%BB%E5%83%8F%E3%81%AE%E3%82%B3%E3%83%94%E3%83%BC) | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Tutorial/README_opencv_c_install.md#%E7%94%BB%E5%83%8F%E3%81%AE%E3%82%B3%E3%83%94%E3%83%BC) |
|  5   |    保存图像    | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Tutorial#%E7%94%BB%E5%83%8F%E3%81%AE%E4%BF%9D%E5%AD%98) | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Tutorial/README_opencv_c_install.md#%E7%94%BB%E5%83%8F%E3%81%AE%E4%BF%9D%E5%AD%98) |
|  6   |    练习问题    | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Tutorial#%E7%B7%B4%E7%BF%92) | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Tutorial/README_opencv_c_install.md#%E7%B7%B4%E7%BF%92%E5%95%8F%E9%A1%8C) |

请在这之后解答提出的问题。问题内容分别包含在各个文件夹中。请使用示例图片`assets/imori.jpg`。在各个文件夹中的`README.md`里有问题和解答。运行答案，请使用以下指令（自行替换文件夹和文件名）：

```python
python answers/answer_@@.py
```

## 问题

详细的问题请参见各页面下的`README`文件（各个页面下滑就可以看见）。
- 为了简化答案，所以没有编写`main()`函数。
- 虽然我们的答案以`numpy`为基础，但是还请你自己查找`numpy`的基本使用方法。

### [問題1 - 10](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10)

| 序号 |            问题             |                            Python                            |                             C++                              |
| :--: | :-------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  1   |          通道替换           | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers/answer_1.py) | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers_cpp/answer_1.cpp) |
|  2   |     灰度化（Grayscale）     | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers/answer_2.py) | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers_cpp/answer_2.cpp) |
|  3   |   二值化（Thresholding）    | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers/answer_3.py) | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers_cpp/answer_3.cpp) |
|  4   |          大津算法           | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers/answer_4.py) | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers_cpp/answer_4.cpp) |
|  5   |          HSV 变换           | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers/answer_5.py) | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers_cpp/answer_5.cpp) |
|  6   |          减色处理           | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers/answer_6.py) | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers_cpp/answer_6.cpp) |
|  7   | 平均池化（Average Pooling） | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers/answer_7.py) | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers_cpp/answer_7.cpp) |
|  8   |   最大池化（Max Pooling）   | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers/answer_8.py) | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers_cpp/answer_8.cpp) |
|  9   | 高斯滤波（Gaussian Filter） | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers/answer_9.py) | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers_cpp/answer_9.cpp) |
|  10  |  中值滤波（Median filter）  | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers/answer_10.py) | [✓](https://github.com/yoyoyo-yo/Gasyori100knock/tree/master/Question_01_10/answers_cpp/answer_10.cpp) |


## TODO

adaptivebinalizatino, poison image blending

## Citation

```bash
@article{yoyoyo-yoGasyori100knock,
    Author = {yoyoyo-yo},
    Title = {Gasyori100knock},
    Journal = {https://github.com/yoyoyo-yo/Gasyori100knock},
    Year = {2019}
}
```

