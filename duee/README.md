## 2020语言与智能技术竞赛：事件抽取任务
1. 任务类型：封闭域事件抽取
2. 描述：句子级事件抽取任务采用DuEE1.0数据集，包含65个已定义好的事件类型约束和1.7万中文句子，分为1.2万训练集，0.15万验证集和0.35万测试集
3. 评估指标: 预测论元与人工标注论元进行匹配，并按字级别匹配F1进行打分，不区分大小写，如论元有多个表述，则取多个匹配F1中的最高值
4. 链接：一个短期一个长期赛事都使用了DuEE数据集
    - [2020语言与智能技术竞赛：事件抽取任务](https://aistudio.baidu.com/aistudio/competition/detail/32/0/task-definition)
    - [千言数据集：信息抽取](https://aistudio.baidu.com/aistudio/competition/detail/46/0/task-definition)
5. Top方案汇总
    - Rank：[视频](https://live.baidu.com/m/media/pclive/pchome/live.html?room_id=4008201814&source=h5pre)
    - Rank12：[github](https://github.com/qiufengyuyi/event_extraction), [知乎](https://zhuanlan.zhihu.com/p/141237763)
6. 样本Demo
    - 输入：包含事件信息的一个或多个连续完整句子
    ```
    {
       "text":"历经4小时51分钟的体力、意志力鏖战，北京时间9月9日上午纳达尔在亚瑟·阿什球场，以7比5、6比3、5比7、4比6和6比4击败赛会5号种子俄罗斯球员梅德韦杰夫，夺得了2019年美国网球公开赛男单冠军。",
       "id":"6a10824fe9c7b2aa776aa7e3de35d45d"
    }
    ```
    - 输出：属于预先定义的事件类型、类型角色的论元结果
    ```
    {
        "id":"6a10824fe9c7b2aa776aa7e3de35d45d",
        "event_list":[
            {
                "event_type":"竞赛行为-胜负",
                "arguments":[
                    {
                        "role":"时间",
                        "argument":"北京时间9月9日上午"
                    },
                    {
                        "role":"胜者",
                        "argument":"纳达尔"
                    },
                    {
                        "role":"败者",
                        "argument":"5号种子俄罗斯球员梅德韦杰夫"
                    },
                    {
                        "role":"赛事名称",
                        "argument":"2019年美国网球公开赛"
                    }
                ]
            },
            {
                "event_type":"竞赛行为-夺冠",
                "arguments":[
                    {
                        "role":"时间",
                        "argument":"北京时间9月9日上午"
                    },
                    {
                        "role":"夺冠赛事",
                        "argument":"2019年美国网球公开赛"
                    },
                    {
                        "role":"冠军",
                        "argument":"纳达尔"
                    }
                ]
            }
        ]
    }
    ```
7. 请自行下载数据到duee/trainsample目录下，[下载地址](https://aistudio.baidu.com/aistudio/competition/detail/46/0/datasets)


