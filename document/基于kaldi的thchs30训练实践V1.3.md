# 基于kaldi的thchs30训练实践V1.3

## 6-24-2019_slip_v1.0

## 6-25-2019_slip_v1.1

## 6-26-2019_slip_v1.2

## 6-28-2019_slip_v1.3



首先在老师给的机器上看一下，到/voice_rec/kaldi/egs/thchs30/s5下，不出所料没有data文件，但是有一个download_and_untar.sh，看了一下代码太多，所以还是直接自己动手丰衣足食。

![1561358182678](C:\Users\HPuser\AppData\Roaming\Typora\typora-user-images\1561358182678.png)

​                                                                     download_and_untar.sh

下载的处理过程：

```
 wget http://cn-mirror.openslr.org/resources/18/data_thchs30.tgz
 wget http://cn-mirror.openslr.org/resources/18/test-noise.tgz
 wget http://cn-mirror.openslr.org/resources/18/resource.tgz
```

完成后mkdir thchs30-openslr并解压到./s5/thchs30-openslr。

修改./s5/cmd.sh为：

```
#export train_cmd=queue.pl       
#export decode_cmd="queue.pl --mem 4G" 
#export mkgraph_cmd="queue.pl --mem 8G"
#export cuda_cmd="queue.pl --gpu 1"
export train_cmd=run.pl 
export decode_cmd="run.pl --mem 4G" 
export mkgraph_cnd="run.pl --mem 8G"                                                     
export cuda_cmd="run.pl --gpu 1"        
```

修改./s5/run.sh为：

```
#n=8      #parallel jobs 
n=4   #change by num of cpuCores 
#thchs=/nfs/public/materials/data/thchs30-openslr                                         
thchs=/home/model/voice_rec/kaldi/egs/thchs30/s5/thchs30-openslr 
```

bash run.sh 便开始训练了。它大概有几个过程：数据准备，monophone单音素训练， tri1三因素训练， trib2进行lda_mllt特征变换，trib3进行sat自然语言适应，trib4做quick，后面就是dnn了 。

![1561366022348](C:\Users\HPuser\AppData\Roaming\Typora\typora-user-images\1561366022348.png)

先到这里，让他跑着。

今天一早上去一看，发现并没有进程。

![1561422645949](C:\Users\HPuser\AppData\Roaming\Typora\typora-user-images\1561422645949.png)

因为如果成功运行结束，thchs30/s5/exp中会有变化，我们可以进去看一下：

![1561424004438](C:\Users\HPuser\AppData\Roaming\Typora\typora-user-images\1561424004438.png)

tri1/final.mdl即为输出的模型，此外graph_word文件夹下面有words.txt,和HCLG.fst，一个是字典，一个是有限状态机。

然后在kaldi/src下

```
make ext
```

编译扩展程序。经过几分钟的编译后，可以在src/onlinebin下看到

![1561424575636](C:\Users\HPuser\AppData\Roaming\Typora\typora-user-images\1561424575636.png)

online-wav-gmm-decode-faster 用来回放wav文件来识别的，online-gmm-decode-faster用来从麦克风输入声音来识别的。

现在配置一个demo：

```bash
cd egs/voxforge/
cp -r ./online_demo/ ../thchs30/#将voxforge下的online_demo cp 到thchs30下
cd ../thchs30/
cd online_demo/
mkdir online-data#创建两个目录
mkdir work
cd online-data/
mkdir audio#创建两个目录
mkdir models
cd models/
mkdir tri1#在models下创建tri1
cd tri1/
cp ../../../../s5/exp/tri1/35.mdl ./#将/thchs30/s5/exp/tri1下的两个文件 cp 到当前目录
cp ../../../../s5/exp/tri1/final.mdl ./
cp ../../../../s5/exp/tri1/graph_word/words.txt ./#将./graph_word下的两个文件 cp 到当前目录
cp ../../../../s5/exp/tri1/graph_word/HCLG.fst ./
```

修改后的文件目录应该是这样的：

```
来自：https://blog.csdn.net/m0_38055352/article/details/82560600tdsourcetag=s_pcqq_aiomsg

online_demo
├── online-data
│   ├── audio
│   │   ├── 1.wav
│   │   ├── 2.wav
│   │   ├── 3.wav
│   │   ├── 4.wav
│   │   ├── 5.wav
│   │   └── trans.txt
│   └── models
│       └── tri1
│           ├── 35.mdl
│           ├── final.mdl
│           ├── HCLG.fst
│           └── words.txt
├── README.txt
├── run.sh
└── work[这个文件夹运行run.sh成功后才会出现]
    ├── ali.txt
    ├── hyp.txt
    ├── input.scp
    ├── ref.txt
    └── trans.txt
```

修改thchs30/online_demo/run.sh：

```bash
:'#Here is changed by slip,we donot need online data
if [ ! -s ${data_file}.tar.bz2 ]; then
    echo "Downloading test models and data ..."
    wget -T 10 -t 3 $data_url;
    if [ ! -s ${data_file}.tar.bz2 ]; then
        echo "Download of $data_file has failed!"
        exit 1
    fi
fi
'
```

```bash
ac_model_type=tri1
#changed by slip,to be our url
```

```bash
online-wav-gmm-decode-faster --verbose=1 --rt-min=0.8 --rt-max=0.85\
--max-active=4000 --beam=12.0 --acoustic-scale=0.0769 \
scp:$decode_dir/input.scp $ac_model/final.mdl $ac_model/HCLG.fst \
$ac_model/words.txt '1:2:3:4:5' ark,t:$decode_dir/trans.txt \
ark,t:$decode_dir/ali.txt $trans_matrix;;
#changed by slip,from model into final.mdl
```

现在便可以开始run了：

```bash
./run.sh    #开始回放识别，即识别.wav文件
./run.sh -test-mode live    #从麦克风识别
```

将B6_390至B6_395共6个文件cp到audio目录下，并且../ run.sh：

![1561429004646](C:\Users\HPuser\AppData\Roaming\Typora\typora-user-images\1561429004646.png)

可以看到虽然错误率有点高，但是还是基本跑通了。

由于前一天第一次训练，没有发现/best_wer 文件，所以又训练了一遍，这次在thchs30/s5/exp/tri1/decode_test_word/scoring_kaldi下发现了该文件，打开：

![1561510409857](C:\Users\HPuser\AppData\Roaming\Typora\typora-user-images\1561510409857.png)

我们可以看到，错误率在36.23%。

因为突然关机，之前写的文档进度没保存，可见及时存档的重要性。。。下面补充一下：

在测试完数据之后，我们准备采取我们自己真实的声音进行识别，看一下识别效率，作为对照，我们准备了两组数据：取自数据集的文本，任意取自互联网的文本。

```
 A19_123.wav.trn
另外 假设 不 麻烦 请 关照 一下 内务 及 星空 花园 原则上 别 让 屋子 变成 鬼屋 就好 了 啦 云云
 A19_124.wav.trn
另外 女单 中国队 还有 韩晶 娜 和 尧 燕 奥运 排名 第六 七位 也 可 与 高手 一 搏
 A19_125.wav.trn
工厂 和 厂房 依 山 而 建 全部 配备 排污 系统 你 见 不到 黑 烟 听不到 噪声 也 看不到 污水
 A19_126.wav.trn
对於 她 在 人 赃 俱 获 的 情况 下 仍 强 辞 夺 理 颇 感 有趣 她 似乎 不知道 绝望 为 何物
 A19_127.wav.trn
要 想 扶 优 汰 劣 首先 要 解决 谁 优 谁 劣 的 问题 要 判定 谁 优 谁 劣 必须 得 有一个 衡量 的 尺子
 A19_128.wav.trn
仅 绘画 而论 齐白石 是 巍巍 昆仑 可 这位 附庸风雅 的 门外汉 连 一块 石头 都 不是
 A19_129.wav.trn
他 患有 风湿 性 腰疼 病 一 粘 潮湿 劳累 就 疼痛 难忍 但 装 井口 又 必须 弯腰 弓 背 钻进 满 是 泥水 的 钻 台 下边 干活
 A19_130.wav.trn
大 花鞋 的 殷勤 与 自信 早已 烟消云散 她 抱着 双臂 冷冷 地 看着 这一切
 A19_131.wav.trn
去年 二月 未 满 十 岁 的 徐敏 又 被 上海 前进 业余 进修 学院 录取 继续 学习 新概念英语 第四册
 A19_132.wav.trn
得到 李公朴 噩耗 闻一多 怒 愤 填 膺 拍案而起 怒斥 反动派 卑鄙无耻

以上为取自数据集的文本
```

kaldi是支持麦克风传入实时识别的，但是由于没有麦克风装置，所以在经过同学录音，然后传入识别后，结果如下：



```
铺天盖地的各种消息是自由球员市场即将开启的重要标志。
不过看似杂乱无序、纠缠成团的局面，却有一个被视为重中之重的线头。
只要沿着这条线抽丝剥茧，所有问题都将明朗化。
手术事宜要待医生对其进行进一步的检查、治疗和评估后再确定。
把这份爱延续下去，这将会是女儿一生中最宝贵的财富。
燕子去了，有再来的时候；杨柳枯了，有再青的时候；桃花谢了，有再开的时候。
盼望着，盼望着，东风来了，春天的脚步近了。
现代散文家朱自清的白话散文对“五四”以后的散文作家产生过一定的影响。
母亲在牌桌上遇见一位太太，她有个女儿，透着聪明伶俐。
随着各地公积金管理政策的优化和完善，不少地方简化了公积金提取手续

以上为取自网络的文本
```

在经过同学录音，然后传入识别后，结果如下：



感谢：

https://blog.csdn.net/m0_38055352/article/details/82560600

https://blog.csdn.net/zhanaolu4821/article/details/88894990