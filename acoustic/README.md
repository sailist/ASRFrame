# 声学模型日志
用于简单介绍里面的各种模型及其大致效果比对

## ABCDNN
A(Attention) B(Batchnorm) C(Conv) D(Deep) NN(nerual network), hhhhh...
### DCNN2D
参考ASRT项目迁移过来的模型，确实有一定效果，在初期帮助我理解CTC解码提供了很大的帮助，但遗憾的是效果确实有限，loss大概能到25。

### DCBNN2D
参考ASRT项目复制过来的模型，包括BatchNorm，同样效果有限，thchs30数据集大概loss只能降到25

### DCBNN1D
将DCBNN2D中的结构不扩维，直接在1D上进行操作，效果出奇的好，清华数据集能够过拟合，loss可以降到3左右，在多数据集上loss大概能降到13左右。

### DCBNN1Dplus
在DCBNN1D的基础上叠加卷积层，目前训练结果来看效果没有DCBNN1D好，可能训练次数不够，欠拟合；也可能层数过多，优化器能力不足。

> 考虑添加残差结构

### DCBANN1D
在DCBNN1D的基础上添加了Attention结构，仍然有一定的效果，但效果比DCBNN1D反而变差了，暂时没有深究原因，可能是因为需要对齐的都聚集在时间步附近，而Attention是在全局范围内寻找对齐要素

## LASModel
《Listen,Attend and Spell》的论文实现（不完全版）


## MAXM
Somiao输入法的声学模型迁移版


## WAVE
wavenet实现