---
title: 【机器学习基础】性能度量(performance measure)
date: 2018-07-25
tags:
categories: ["机器学习笔记"]
mathjax: true
---

&emsp; 性能度量(performance measure)是衡量模型泛化能力的评价标准。在对比不同模型的performance时，使用不同的 performance measure 往往会导致不同的评判结果，这意味着模型的“好坏”是相对的，什么样的模型是好的，不仅取决于算法和数据，还取决于任务需求 [1]。
<!-- more -->

回归任务最常用的 performance measure 是“均方误差”（mean squared error）
$$E(f;D) = \frac{1}{m} \sum^m_{i=1}(f(\boldsymbol{x}_i) - y_i)^2 \ .$$

本节主要介绍分类任务中常用的 performance measure.


# 错误率与精度
&emsp; 错误率（error rate）是分类错误的样本数占样本总数的比例，精度（accuracy）则是分类正确的样本数占样本总数的比例. 对样例集D，分类错误率定义为
$$E(f;D) = \frac{1}{m} \sum^m_{i=1} \mathbb I(f(\boldsymbol{x}_i) \not ={y_i})$$

精度定义为
$$E(f;D) = \frac{1}{m} \sum^m_{i=1} \mathbb I(f(\boldsymbol{x}_i) = {y_i}) = 1 - E(f;D)$$


# 查准率、查全率与$F_{\beta}$
&emsp; error rate 和 accuracy 虽然常用，但并不能满足所有任务需求。以挑西瓜问题为例，我们希望从一车西瓜中挑出好瓜，如果我们此时更关心“挑出的西瓜中好瓜所占的比例”（实际上就是Precision），或者“所有好瓜中有多少比例被挑了出来”（实际上就是Recall），这时候error rate 和 accuracy 显然就不能满足需求了。

&emsp; 对于二分类问题，根据真实类别和学习器预测类别的组合划分，有混淆矩阵(confusion matrix)
| 真实情况 \ 预测结果 | 正例 | 负例 | 合计 |
| --- | --- | --- | --- |
| 正例 | True Positive(TP) | False Negtive(FN) | P(真的为正例的所有样本) |
| 负例 | False Positive(FP) | True Negtive(TN) | N(真的为负例的所有样本) |
| 合计 | P'(预测为正例的所有样本) | N'(预测为负例的所有样本) | P+N=P'+N' |

查准率（Precison）和查全率（Recall）分别定义为
$$P = \frac{TP}{TP+FP} = \frac{TP}{P'}$$
$$R = \frac{TP}{TP+FN} = \frac{TP}{P}$$

&emsp; 查准率可以认为是”宁缺毋滥”，适合对准确率要求高的应用，例如商品推荐，网页检索等。查全率可以认为是”宁错杀一百，不放过1个”，适合类似于检查走私、逃犯信息等。

&emsp; 根据学习器的预测结果对样例进行排序，排名越靠前则越有可能是正例的样本，按此顺序逐个把样本作为正例，剩下的都预测为负例，则每次可以计算出当前的查全率和查准率，以查全率为横轴可得**查准率-查全率曲线**，简称**P-R曲线**，显示该曲线的图称为**P-R图**.

&emsp; 在进行比较时，若一个学习器A的P-R曲线被另一个学习器B的P-R曲线完全“包住”，则可断言B的性能优于A。但是如果两个学习器的P-R曲线发生交叉，则难以一般性地断言两者孰优孰劣，只能在具体的查准率和查全率条件下进行比较。然而，在很多情况下，人们往往仍希望把学习器之间比个高低，这时候一个比较合理的判据是比较P-R曲线下的面积大小，它在一定程度上表征了学习器在查准率和查全率上取得相对“双高”的比例。但这个值往往不太容易估算，因此，人们设计了一些综合考虑查准率、查全率的度量 [1]。（其实就是 Precision 和 Recall 率的权衡）

## 平衡点
平衡点（Break-Event Point, 简称BEP）就是这样一个度量，它是“Precison = Recall”时的取值，我们认为平衡点越靠近（1,1）则性能越好。

## $F_{\beta}$度量
&emsp; 但BEP还是过于简单了些，例如下面两个模型哪个综合性能更优？
|   | Precision | Recall |
| --- | --- | --- |
| A模型 | 80% | 90% |
| B模型 | 90% | 80% |

为了解决这个问题，人们提出了 $F_{\beta}$ 度量。$F_{\beta}$ 能够表达出对 Precision/Recall 的不同偏好，其物理意义就是将 Precision 和 Recall 这两个得分值(score)合并为一个值，在合并的过程中，Recall 的权重是 Precision 的 $\beta$ 倍 [2]，即 $W_R = \beta \cdot W_P$， 则
$$F_{\beta} = (1+\beta^2) \cdot \frac{P \times R}{(\beta^2 \times P) + R} \qquad \beta > 0$$

当 $\beta = 1$ 时，$F_{\beta}$ 退化为标准的 $F_{1}$ 度量，此时Precision 和 Recall 同等重要；

当 $\beta > 1$ 时，则 Recall 更重要，当 $\beta < 1$ 时，则 Precision 更重要。

> $F_{1}$ 是基于 Precision 和 Recall 的调和平均（harmonic mean）定义的：
> $$F_{1} = 2 \cdot \frac{P \times R}{P + R}$$
> $$\frac{1}{F_{1}} = \frac{1}{2} \cdot \frac{P + R}{P \times R} = \frac{1}{2} \cdot (\frac{1}{P} + \frac{1}{R})$$
> 
> $F_{\beta}$ 则是加权调和平均：
> $$\frac{1}{F_{\beta}} = \frac{1}{1+\beta^2} \cdot (\frac{1}{P} + \frac{\beta^2}{R})$$
> 
> 与算术平均（$\frac{P+R}{2}$）和几何平均相比（$\sqrt{P \times R}$）相比，调和平均更重视较小值。


## $macroP、macroR、macroF_1$
&emsp; 很多时候我们有多个二分类 confusion matrix，例如进行多次 Train/Test，每次得到一个 confusion matrix；或是在多个数据集上进行 Train/Test，希望估计算法“全局”的 performance；甚或是执行多分类任务，每**两两类别的组合**都对应一个 confusion matrix；…… 总之，我们希望在 n 个二分类 confusion matrix 上综合考察查准率和查全率. [1]

&emsp; 一种直接的做法是先在各 confusion matrix 上分别计算出 Precision 和 Recall，记为 $(P_{1}, R_{1}), (P_{2}, R_{2}), ... , (P_{n}, R_{n})$，再计算平均值，这样就得到“宏查准率”（macro-P）、“宏查全率”（macro-R），以及相应的“宏$F_1$”（macro-$F_1$）:
$$macroP = \frac{1}{n} \sum^n_{i=1} P_i$$
$$macroR = \frac{1}{n} \sum^n_{i=1} R_i$$
$$macroF_1 = \frac{2 \times {macroP} \times macroR}{macroP + macroR}$$


## $microP、microR、microF_1$
&emsp; 还以先将各 confusion matrix 的对应元素进行平均，得到 TP、FP、TN、FN 的平均值，再基于这些平均值计算出“微查准率”（micro-P）、“微查全率”（micro-R），以及相应的“微$F_1$”（micro-$F_1$）:
$$microP = \frac{\overline{TP}}{\overline{TP} + \overline{FP}}$$
$$microR = \frac{\overline{TP}}{\overline{TP} + \overline{FN}}$$
$$microF_1 = \frac{2 \times {microP} \times microR}{microP + microR}$$


# ROC 与 AUC
&emsp; 机器学习中的很多模型对于分类问题的预测结果大多是概率，即属于某个类别的概率，如果计算准确率的话，就要把概率转化为类别，这就需要设定一个阈值，概率大于某个阈值的属于一类，概率小于某个阈值的属于另一类，而阈值的设定直接影响了学习器的泛化能力。

&emsp; 与P-R曲线类似，我们根据学习器的预测结果(概率)对样例进行排序，最可能是正例的排在最前面，最不可能是正例的排在最后面，按此顺序逐个把样本作为正例(逐个设定阈值)进行预测，计算每个阈值下的“真正例率”（True Positive Rate，简称TPR）、“假正例率”（False Positive Rate，简称FPR）:
$$TPR = \frac{TP}{TP + FN} = \frac{TP}{P}$$
$$FPR = \frac{FP}{TN + FP} = \frac{FP}{N}$$
TPR 实际上就是 Recall，表示所有的正样本中有多少被召回了；而 FPR 表示所有的负样本中有多少被误分为正样本了。

上面说到，每个阈值会得到一组$(FPR, \ TPR)$，那么 n 组阈值就会得到 n 组 $(FPR_0, \ TPR_0), \ (FPR_1, \ TPR_1), \ ... \ , \ (FPR_n, \ TPR_n)$，使用这 n 组$(FPR_i, \ TPR_i)$ 以 FPR 为横轴，以 TPR 为纵轴作图，就得到了“**ROC曲线**”，显示 ROC 曲线的图称为 **ROC图**。

显然，ROC图中的对角线对应于“随机猜测”模型。

与P-R图类似，若一个学习器的ROC曲线另一个学习器的曲线完全包住，则可断言后者的性能优于前者；若两个学习器的ROC曲线发生交叉，则难以一般性地断言两者孰优孰劣。此时如果一定要进行比较，则较为合理的判据是比较ROC曲线下的面积，即AUC（Area Under ROC Curve）.


# P-R曲线与ROC曲线的选择
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/ml/PR_ROC.jpg" width = 70% height = 70% />
</div>

本段来自文献[2]

**如何选择呢？**

- 在很多实际问题中，正负样本数量往往很不均衡。比如，计算广告领域经常涉及转化率模型，正样本的数量往往是负样本数量的1/1000，甚至1/10000。若选择不同的测试集，P-R曲线的变化就会非常大，而ROC曲线则能够更加稳定地反映模型本身的好坏。所以，ROC曲线的适用场景更多，被广泛用于排序、推荐、广告等领域。

- 但需要注意的是，选择P-R曲线还是ROC曲线是因实际问题而异的，如果研究者希望更多地看到模型在特定数据集上的表现，P-R曲线则能够更直观地反映其性能。

- PR曲线比ROC曲线更加关注正样本，而ROC则兼顾了两者。

- AUC越大，反映出正样本的预测结果更加靠前。（推荐的样本更能符合用户的喜好）

- 当正负样本比例失调时，比如正样本1个，负样本100个，则ROC曲线变化不大，此时用PR曲线更加能反映出分类器性能的好坏。这个时候指的是两个分类器，因为只有一个正样本，所以在画auc的时候变化可能不太大；但是在画PR曲线的时候，因为要召回这一个正样本，看哪个分类器同时召回了更少的负样本，差的分类器就会召回更多的负样本，这样precision必然大幅下降，这样分类器性能对比就出来了。



# 参考文献
[1] 机器学习.周志华 > 2.3 性能度量 \
[2] [一文详尽混淆矩阵、准确率、精确率、召回率、F1值、P-R 曲线、ROC 曲线、AUC 值、Micro-F1 和 Macro-F1](https://blog.csdn.net/weixin_37641832/article/details/104434509?fps=1&locationNum=2) \
[3] 代码参考 > [sklearn ROC曲线使用](https://blog.csdn.net/hfutdog/article/details/88079934)

