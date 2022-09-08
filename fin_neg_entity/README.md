## 金融信息负面及主体判定


1. 任务类型：实体关联的情感分类问题 
2. 描述：训练集数据量1万条，测试集数据量1万条。给定一条金融文本和文本中出现的金融实体列表，完成一下两个任务
    - 负面信息判定：判定该文本是否包含金融实体的负面信息。如果该文本不包含负面信息，或者包含负面信息但负面信息未涉及到金融实体，则负面信息判定结果为0。
    - 负面主体判定：如果任务1中包含金融实体的负面信息，继续判断负面信息的主体对象是实体列表中的哪些实体。
3. 评估指标:采 面判定指标F1​、主体判定指标F1​和任务整体得分F1三个评价指标
4. 链接：https://www.datafountain.cn/competitions/353
5. Top方案汇总
    - Top1: [github](https://github.com/xiong666/ccf_financial_negative) , [zhihu](https://zhuanlan.zhihu.com/p/99222193)
    - Top3: [github](https://github.com/Chevalier1024/CCF-BDCI-ABSA), [zhihu](https://zhuanlan.zhihu.com/p/97900951)
    - Top5: [github](https://github.com/rebornZH/2019-CCF-BDCI-NLP) 
 6. 样本Demo

 ![](https://files.mdnice.com/user/8955/1486ab47-8a3d-4451-a07f-802c2a007dc7.png) 