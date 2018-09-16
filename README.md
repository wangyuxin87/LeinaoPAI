## LeinaoPAI


### 欢迎大家分享更多的使用案例

| 序号 | 名称                                                       | 摘要                                                         | 提供者      | 更新时间  | 备注 |
| ---- | ---------------------------------------------------------- | ------------------------------------------------------------ | ----------- | --------- | ---- |
| 1    | [tensorflow-distributed](./example/tensorflow-distributed) | 提交跨机器的任务进行TensorFlow的分布式训练，以mnist数据集分类为例 | feng257     | 2018.8.14 |      |
| 2    | [pytorch-imagenet](./example/pytorch-imagenet)                                           | 在pytorch框架下，使用平台的ImageNet数据集，训练resnet50      | feng257     | 2018.9.15 |      |
| 3    | [ppchallenge2018](./example/ppchallenge2018)                                            | 在平台上支持“ [基于深度学习的图像压缩后处理竞赛](https://www.cnbita.com/views/activity-detail.html?activityId=_ad7d2bc8117f449893f007da80d8072b)”的后台代码    | xwzheng1020 | 2018.9.16 |      |


### 欢迎大家将镜像的dockerfile分享出来

当有编译镜像需求时，提交pull request到本项目，要求：
1. 统一放置在images目录下
2. 每个镜像一个子目录，目录名称为"镜像名称.tag"，如 ”ubuntu.16.04-ssh"
3. 如果镜像编译过程中需要使用一些大的文件，不要提交到本项目，而是提供一个下载这些文件的脚本
4. 同时提交readme文件



### 友情链接

[彼塔社区](https://www.cnbita.com)

[OpenPAI](https://github.com/microsoft/pai)

