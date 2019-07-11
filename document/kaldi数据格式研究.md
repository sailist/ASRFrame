# kaild数据格式研究_slip_v1.0



进入/kaldi/egs/yesno/s5/data/train_yesno/TEXT可以看到如下数据：

![1561186729266](C:\Users\HPuser\AppData\Roaming\Typora\typora-user-images\1561186729266.png)

经过分析不难发现，no和yes分别对应0和1。

查看其他文件，结果如下：

![1561191668072](C:\Users\HPuser\AppData\Roaming\Typora\typora-user-images\1561191668072.png)

![1561191702490](C:\Users\HPuser\AppData\Roaming\Typora\typora-user-images\1561191702490.png)

观察[kaldi文档](https://kaldi-asr.org/doc/kaldi_for_dummies.html)我们可以知道：

### 对于声学数据：

#### 1.spk2gender

此文件包含了发言人的性别，格式为：

speaker1     gender1

speaker2     gender2

...

其中speaker为发言人的代号 gender为f（女性）或m（男性）

#### 2.wav.scp

此文件包含了音频的id与其音频文件的映射，格式为：

speaker1_i_j_k      /home/{user}/kaldi/egs/digits/digits_audio/train/speaker1/i_j_k.wav

ijk表示其语音内容（个人推测，因为在他给的例子中，是让一个人说三个数字作为一个音频）

除此之外，我们也可推测出他的文件系统是按发言人（speaker）来组织的，在数据集内以发言人的代号创建相应的文件夹，并且在该文件夹下保存其音频。

#### 3.text

此文件包含音频id与其语音内容的匹配关系，格式为：

dad_4_4_2    four four two

#### 4.utt2spk

此文件包含音频id与发言人的匹配关系，格式为：

speaker1_i_j_k     speaker1

#### 5.corpus.txt

这个文件的目录为kaldi/egs/digits/data/local，他包含了每一条会发生在ASR系统中的转录，格式如下：

```
one two five
six eight three
four four two
# and so on...
```

（例子中是三个数字的发音）

### 对于语言数据：

#### 1.lexicon.txt

此文件包含了每个单词与其音素的匹配，格式为：

```
!SIL sil
<UNK> spn
eight ey t
five f ay v
four f ao r
nine n ay n
one hh w ah n
one w ah n
seven s eh v ah n
six s ih k s
three th r iy
two t uw
zero z ih r ow
zero z iy r ow
```

（取自例子）

#### 2.nonsilence_phones.txt

此文件包含了非沉默（nonsilence，我不懂，直译的。。应该是指一类特殊的）音素，格式为：

```
ah
ao
ay
eh
ey
f
hh
ih
iy
k
n
ow
r
s
t
th
uw
w
v
z
```

（取自例子）

#### 3.silence_phones.txt

此文件包含了沉默（silence，同样直译）音素，格式为：

```
sil
spn
```

（取自例子）

#### 4.optional_silence.txt

此文件包含可选沉默（直译）音素，格式为：

```
sil
```

（取自例子）

##### 猜测：这部分几个文件的区分理由应该是发音方式，拼音的话应该与这个例子有不同之处。

##### 以下是原文：

> ## Acoustic data
>
> Now you have to create some text files that will allow Kaldi to communicate with your audio data. Consider these files as 'must be done'. Each file that you will create in this section (and in [Language data](https://kaldi-asr.org/doc/kaldi_for_dummies.html#kaldi_for_dummies_language) section as well) can be considered as a text file with some number of strings (each string in a new line). These strings need to be sorted. If you will encounter any sorting issues you can use Kaldi scripts for checking (`utils/validate_data_dir.sh`) and fixing (`utils/fix_data_dir.sh`) data order. And for your information - `utils` directory will be attached to your project in [Tools attachment](https://kaldi-asr.org/doc/kaldi_for_dummies.html#kaldi_for_dummies_tools) section.
>
> ## Task
>
> In `kaldi/egs/digits` directory, create a folder `data`. Then create `test` and `train` subfolders inside. Create in each subfolder following files (so you have files named in **the same way in test and train subfolders but they relate to two different datasets** that you created before):
>
> a.) `spk2gender` 
> This file informs about speakers gender. As we assumed, 'speakerID' is a unique name of each speaker (in this case it is also a 'recordingID' - every speaker has only one audio data folder from one recording session). In my example there are 5 female and 5 male speakers (f = female, m = male).
>
> **Pattern:** <speakerID> <gender>
>
> ```
> cristine f
> dad m
> josh m
> july f
> # and so on...
> ```
>
> b.) `wav.scp` 
> This file connects every utterance (sentence said by one person during particular recording session) with an audio file related to this utterance. If you stick to my naming approach, 'utteranceID' is nothing more than 'speakerID' (speaker's folder name) glued with *.wav file name without '.wav' ending (look for examples below).
>
> **Pattern:** <uterranceID> <full_path_to_audio_file>
>
> ```
> dad_4_4_2 /home/{user}/kaldi/egs/digits/digits_audio/train/dad/4_4_2.wav
> july_1_2_5 /home/{user}/kaldi/egs/digits/digits_audio/train/july/1_2_5.wav
> july_6_8_3 /home/{user}/kaldi/egs/digits/digits_audio/train/july/6_8_3.wav
> # and so on...
> ```
>
> c.) `text` 
> This file contains every utterance matched with its text transcription.
>
> **Pattern:** <uterranceID> <text_transcription>
>
> ```
> dad_4_4_2 four four two
> july_1_2_5 one two five
> july_6_8_3 six eight three
> # and so on...
> ```
>
> d.) `utt2spk` 
> This file tells the ASR system which utterance belongs to particular speaker.
>
> **Pattern:** <uterranceID> <speakerID>
>
> ```
> dad_4_4_2 dad
> july_1_2_5 july
> july_6_8_3 july
> # and so on...
> ```
>
> e.) `corpus.txt` 
> This file has a slightly different directory. In `kaldi/egs/digits/data` create another folder `local`. In `kaldi/egs/digits/data/local` create a file `corpus.txt` which should contain every single utterance transcription that can occur in your ASR system (in our case it will be 100 lines from 100 audio files).
>
> **Pattern:** <text_transcription>
>
> ```
> one two five
> six eight three
> four four two
> # and so on...
> ```
>
> ## Language data
>
> This section relates to language modeling files that also need to be considered as 'must be done'. Look for the syntax details here: [Data preparation](https://kaldi-asr.org/doc/data_prep.html) (each file is precisely described). Also feel free to read some examples in other `egs` scripts. Now is the perfect time.
>
> ## Task
>
> In `kaldi/egs/digits/data/local` directory, create a folder `dict`. In `kaldi/egs/digits/data/local/dict` create following files:
>
> a.) `lexicon.txt` 
> This file contains every word from your dictionary with its 'phone transcriptions' (taken from `/egs/voxforge`).
>
> **Pattern:** <word> <phone 1> <phone 2> ...
>
> ```
> !SIL sil
> <UNK> spn
> eight ey t
> five f ay v
> four f ao r
> nine n ay n
> one hh w ah n
> one w ah n
> seven s eh v ah n
> six s ih k s
> three th r iy
> two t uw
> zero z ih r ow
> zero z iy r ow
> ```
>
> b.) `nonsilence_phones.txt` 
> This file lists nonsilence phones that are present in your project.
>
> **Pattern:** <phone>
>
> ```
> ah
> ao
> ay
> eh
> ey
> f
> hh
> ih
> iy
> k
> n
> ow
> r
> s
> t
> th
> uw
> w
> v
> z
> ```
>
> c.) `silence_phones.txt` 
> This file lists silence phones.
>
> **Pattern:** <phone>
>
> ```
> sil
> spn
> ```
>
> d.) `optional_silence.txt` 
> This file lists optional silence phones.
>
> **Pattern:** <phone>
>
> ```
> sil
> ```

