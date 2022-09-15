## 金融信息负面及主体判定

1. 任务类型：实体关联的情感分类问题 
2. 描述：训练集数据量1万条，测试集数据量1万条。给定一条金融文本和文本中出现的金融实体列表，完成一下两个任务
    - 负面信息判定：判定该文本是否包含金融实体的负面信息。如果该文本不包含负面信息，或者包含负面信息但负面信息未涉及到金融实体，则负面信息判定结果为0。
    - 负面主体判定：如果任务1中包含金融实体的负面信息，继续判断负面信息的主体对象是实体列表中的哪些实体。
3. 评估指标: 负面判定指标F1​、主体判定指标F1​，整体任务F1=0.4负面F1+0.6实体F1
4. 链接：https://www.datafountain.cn/competitions/353
5. Top方案汇总
    - Top1: [github](https://github.com/xiong666/ccf_financial_negative) , [zhihu](https://zhuanlan.zhihu.com/p/99222193)
    - Top3: [github](https://github.com/Chevalier1024/CCF-BDCI-ABSA), [zhihu](https://zhuanlan.zhihu.com/p/97900951)
    - Top5: [github](https://github.com/rebornZH/2019-CCF-BDCI-NLP) 
 6. 样本Demo
 ![](https://files.mdnice.com/user/8955/1486ab47-8a3d-4451-a07f-802c2a007dc7.png) 
 7. 请自行按链接下载数据到./fin_neg_entity/trainsample目录下
 8. 注意；2021年之前DataFoundation上的比赛评测程序都缺失，因此无法再提交测试机评估，这里用相同seed分割的valid数据集来对比效果
 9. 尝试了5种方案

| 方案    | F1_entity|  F1_sentence   | F1     |
| --- | --- | --- | --- |
|Format1：双输入，1为待预测实体，2为title+text，伴随实体用[O]替换|  93.8%   |94.5%| 94.1% |
|Format2：单输入，title+text, 用[E]标记待预测实体，伴随实体用[O]替换 | 93.5%  | 94.6%  |93.9%  |
|Format3：双输入，1为伴随实体拼接，2title+text，带预测实体用[E]标记  | 94.5% | 95% |  94.7%|
|Format4:  Format3基础上加入多任务同时学习实体+句子负面  | 94.8%    | 95.5%     |95.1%|
|Format5：Format4基础上加入TAPT| 95.0%| 95.6%|95.2%    |
