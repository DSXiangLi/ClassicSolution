## 互联网金融新实体发现

1. 任务类型：实体识别 & 新词发现
2. 描述：训练集数据量1万条，测试集数据量1万条。每条数据包括标识号（id）、文本标题（title）、文本内容（text）、未知实体列表（unknownEntities）
3. 评估指标: 考察未知实体的识别,识别实体的Micro F1作为评估指标
4. 链接：https://www.datafountain.cn/competitions/361
5. Top方案汇总
    - Top1: [github](https://github.com/ChileWang0228/Deep-Learning-With-Python/tree/master/chapter8) 
    , [知乎](https://zhuanlan.zhihu.com/p/100884995)
    - Top4: [github](https://github.com/rebornZH/2019-CCF-BDCI-NLP)
    - Top5: [github](https://github.com/light8lee/2019-BDCI-FinancialEntityDiscovery) ,[论文](https://github.com/light8lee/2019-BDCI-FinancialEntityDiscovery/blob/master/resources/paper.pdf)
 6. 样本Demo
 7. 请自行按链接下载数据到./fin_new_entity/trainsample目录下
 8. 注意；2021年之前DataFoundation上的比赛评测程序都缺失，因此无法再提交测试机评估，这里用相同seed分割的valid数据集来对比效果
