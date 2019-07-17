### 实现说明

PyTorch实现，基于[huggingface](https://github.com/huggingface/pytorch-pretrained-BERT)的工作，PyTorch才是世界上最屌的框架，逃。

### 实现参考

![img1](http://wx2.sinaimg.cn/mw690/aba7d18bgy1g47p0g5ln3j210n0drtas.jpg)


### 代码说明

（1）主要修改：[modeling.py](https://github.com/zhpmatrix/BERTem/blob/master/pytorch_pretrained_bert/modeling.py)

output representation: **BertForSequenceClassification**

input representation:  **BertEmbeddings**

input和output都实现了多种策略，可以结合具体的任务，找到最佳的组合。


（2）非主要实现：examples下的关于classification的文件

（3）服务部署：基于Flask，可以在本地开启一个服务。具体实现在[tacred\_run\_infer.py](https://github.com/zhpmatrix/BERTem/blob/master/examples/tacred_run_infer.py)中。

（4）代码仅供参考，不提供数据集，不提供预训练模型，不提供训练后的模型（希望理解吧）。

（5）相关工作可以参考[我的博客-神经关系抽取](https://zhpmatrix.github.io/2019/06/30/neural-relation-extraction/)，可能比这个代码更有价值一些吧。


### 实现结果：

 数据集TACRED上的结果：

|模型序号|输入类型|输出类型|指标类型|P|R|F1|备注|
|------|------|------|------|------|------|------|------|
|0|entity marker|sum(entity start)|micro|**0.68**|**0.63**|**0.65**|**base-model**,lr=3e-5,epoch=3|
||||macro|**0.60**|**0.54**|**0.55**|
|1|entity marker|sum(entity start)|micro|**0.70**|**0.62**|**0.65**|**large-model**,lr=3e-5,epoch=1|
||||macro|**0.63**|**0.52**|**0.55**|
|-1|None|None|micro|**0.69**|**0.66**|**0.67**|手误之后，再也找不到了，尴尬|||
||||macro|**0.58**|**0.50**|**0.53**||||


数据集SemEval2010 Task 8上的结果：

|模型序号|输入类型|输出类型|指标类型|P|R|F1|备注|
|------|------|------|------|------|------|------|------|
|0|entity marker|maxpool(entity emb)+relu|micro|**0.86**|**0.86**|**0.86**|bert-large|
||||macro|**0.82**|**0.83**|**0.82**||||


### 混合精度加速结果

在具体任务上，延续之前的setting，将train和dev合并共同作为新的train集，test集不变。在fp32
和fp16的两种setting下，比较相同batch\_size下，一个epoch的用时或者每个迭代的用时。

|比较方面|fp32|fp16|备注|
|------|------|------|------|
|训练阶段|1.04it/s|4.41it/s|12.76it/s（独占显卡）|
|推断阶段|4.14it/s|8.63it/s||
|测试集指标|0.65/0.55|0.64/0.53|格式：micro/macor|
|模型大小|421M|212M||
