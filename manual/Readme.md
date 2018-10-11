### 一、概述

![](http://img.cnbita.com/_f612c7ed7f0e4e338bc77c3575d23715集群概览图.png)
通过Web页面，用户可以向集群提交计算任务，实时查看输出日志。

集群主要提供GPU算力，所有任务都运行在Docker容器中，支持多机多卡运行。

集群可提交各类深度学习任务，也可提交自定义Docker镜像，以执行更多类型的计算任务。

更多详细信息可以关注开源项目[PAI](http://www.github.com/microsoft/pai)

### 二、使用过程

![](http://img.cnbita.com/_3f127e723a314bfa80c519d967ef4dae使用过程图.png)
### 2.1 注册与登录
1. 用浏览器访问[https://202.38.95.226:6443](https://202.38.95.226:6443)，推荐浏览器为chrome
2. 用户需要先在平台注册，方可提交任务。  **注册所需的邀请码，请发送邮件并注明个人信息到xwzheng@leinao.ai**
3. 原先的[https://202.38.95.226:7443](https://202.38.95.226:7443)已经和[https://202.38.95.226:6443](https://202.38.95.226:6443)合并，即这个网址指向同一个网站


### 2.2 调试与提交任务
1. 代码和数据集准备

- 每个在集群中运行的任务都会自动挂载`/userhome`目录，这是用户专属目录，可以存放持久化数据，即本次任务写入该目录（及其子目录）的数据在下次任务启动时仍然存在。
- 用户的代码和数据可以通过[文件中心](https://202.38.95.226:6443/fileCenter.html)上传，`文件中心`中的根目录等同于任务运行时的`/userhome`
- 平台提供了常见的[数据集](https://202.38.95.226:6443/dataset.html)，在`/gdata`目录下 ，如[imagenet数据集](https://202.38.95.226:6443/dataset.html?dataSet=1000000029)的存放路径为`/gdata/imagenet-1000`
- 若用户所需的数据集在平台未提供，可以自行通过`文件中心`上传
- 欢迎在[彼塔社区](https://www.bitahub.com/views/communicate.html)中提更多的数据集预下载需求

2. docker镜像准备 

  - 平台提供部分镜像，如[deepo](https://github.com/ufoym/deepo)等，更多的镜像会陆续增加
  - 若需要自定义镜像，可以参考[dockerhub](https://docs.docker.com/docker-hub/builds/)教程，打包镜像后上传.
 - 欢迎将常用的镜像分享给大家，具体方式可以参考[这里](https://github.com/leinao/leinaopai)，管理员会及时编译并更新到[镜像列表](https://202.38.95.226:6443/imageset.html)

3. 从页面提交任务
  - 在往集群传输数据和代码之后，可在“提交任务”页面中，选择任务所需的软硬件配置  

    ![参数填写](http://img.cnbita.com/_d1a0d051250c489cb1f1953b712066b4参数填写.png) 

  - 任务提交成功之后，集群会按任务提交先后顺序调度运行。 
4. 调试任务
  - 有时任务所要运行的命令比较复杂，需要ssh登录进行调试，那么**在填写gpuType时，务必将其设置为debug**，之后任务会被调度到调试节点上运行，具有**公网IP**和端口。![](http://img.cnbita.com/_a45e7c63790a4192b6f28c77f6b566c4查看SSH Info.png)  
  - 也许你还有可能需要对远程执行的代码进行单步调试，那么可以参考[使用 Visual Studio Code 进行远程调试](http://www.cnbita.com/views/article-detail.html?articleId=_210bd928f7f64b65bd0aa6658b0697b4)和[使用PyCharm调试远程服务器上的python代码](http://www.cnbita.com/views/article-detail.html?articleId=_15061b9c318d4eba913e51bee43dee64)
  - 既然可以通过ssh登录到任务中，那么也可以使用scp工具把本地的数据传输到`/userhome`目录下。这是除`文件中心`外的另一种数据传输方法。

### 2.3 查看任务

1. 可在任务列表页查看个人历史的所有任务

2. 更具体地，可以进入任务详情页面，查看日志和资源使用情况

![](http://img.cnbita.com/_bf5259311fbc42779b228a0762cccb96_67d37572338f44bfbb2c757bcb679a96任务详情.png)

### 2.4 下载结果  
1. 请将模型等重要结果保存在`/userhome`路径下，只有这个路径时用户可持久化的。  
2. 结果的下载方式同数据上传，既可以通过`文件中心`，也可以使用scp。

### 三、典型案例

用户也可以通过导入json脚本的方式输入所需要的软硬件配置：

![](http://img.cnbita.com/_cdc9c591f9774f5391fdc1b8e1f87a38_421a43b21fc740f5ad4f9ef37aa33122脚本提交任务.png)

以下的例子将直接用json脚本作说明。

### 3.1 Hello GPU Cluster

编写[任务提交脚本](https://raw.githubusercontent.com/Leinao/LeinaoPAI/master/example/hello-gpu-cluster/hello_gpu_cluster.json)

```json
{
  "jobName": "jobName_test",	
  "image": "ubuntu:16.04",
  "taskRoles": [
    {
      "name": "hello_gpu_cluster",
      "taskNumber": 1,
      "cpuNumber": 1,
      "memoryMB": 512,
      "gpuNumber": 0,
      "command": "cd /userhome; sleep 30; mkdir -p output; date > output/out.txt; echo finished"
    }
  ]
}
```

1. 该示例表示任务在启动后会依次进行以下操作：
    1）进入个人目录/userhome; 
    2）进行一些耗时的运算（这里用sleep代替）；
    3）输出结果到output目录中（这里用date代替）。
2. 在这过程中，集群做了这些动作：

![](http://img.cnbita.com/_7ca7271b56d34b27a2a6ca7449bfd264_075009ca02814afda8007095319748e5集群运行过程示意图.png)


### 3.2 Pytorch训练Imagenet
编写[任务提交脚本](https://raw.githubusercontent.com/Leinao/LeinaoPAI/master/example/pytorch-imagenet/job-4gpu.json)
```json
{
    "jobName": "pytorch-imagenet-4gpu-",
    "image": "10.11.3.8:5000/pai-images/deepo:v2.0",
    "gpuType": "",
    "retryCount": 0,
    "taskRoles": [
        {
            "name": "run_on_4_gpu",
            "taskNumber": 1,
            "cpuNumber": 8,
            "memoryMB": 65536,
            "shmMB": 8192,
            "gpuNumber": 4,
            "command": "cd /userhome/; if [ ! -d LeinaoPAI ]; then  git clone https://github.com/Leinao/LeinaoPAI.git; else cd LeinaoPAI; git pull origin master; fi; mkdir -p checkpoints; python /userhome/LeinaoPAI/example/pytorch-imagenet/main.py --batch-size 256 --gpu_num 4"
        }
    ]
}
```

### 3.3 利用TensorFlow做分布式图像分类实验
TODO



## 四、问题反馈

1. 优先选择[彼塔社区](https://www.bitahub.com/views/communicate.html)中反馈，社区和集群的账号是通用的。

2. 也可以在微信群中提问题。
  ![](http://img.cnbita.com/_4978e26624064dcf8087a07ef5a004f7weixin_20180919155503.png)