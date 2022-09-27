## 应用类型识别挑战赛

1. 任务类型：文本多分类任务 + 文本脱敏
2. 描述：赛题数据包含4个特征字段：id, name, description, label，从中抽取约4000条作为训练集，约2000条作为测试集，同时会对文字信息进行脱敏
3. 评估指标：micro F1
4. 链接：https://challenge.xfyun.cn/topic/info?type=scene-division
5. Top方案【待补充】
6. 数据Demo 
 ![](https://files.mdnice.com/user/8955/092c5570-8f3b-429e-ac31-a1ad07b8292a.png)
7. 请自行按链接下载数据到./iflytek_app/trainsample目录下
8. 方案：基于脱敏文本训练char2vec，分词器Phraser和基于分词器的word2vec，第一次用pytorch基本是在搬运之前TF的代码，不排除bug良多哈哈~
| 方案    |   | F1     |
| --- |  --- |
|TextCNN+char2vec输入| 0.75292 |
|Fasttext+char2vec输入|   0.75569 |
|TextCNN+char2vec+word2vec输入|	0.77068	|
|TextCNN+char2vec+word2vec输入+FGM|0.77124|
|TextCNN+char2vec+word2vec输入+Temporal|0.77401|
|TextCNN+char2vec+word2vec输入+输入增强|0.77457|
|TextCNN+char2vec+word2vec输入+mixup | 0.78179|
