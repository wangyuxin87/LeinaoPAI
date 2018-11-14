## 类脑计算平台使用样例-MNIST数据训练

本例程给出一个简单的使用类脑计算平台的训练MNIST数据集的样例，其中包括了tensorflow, pytorch和mxnet三种常用框架。针对pytorch框架，推荐使用[ignite](https://pytorch.org/ignite/)包。训练中需要使用的MINST数据已经包含在了平台数据集资源中，所以本例程中MNIST数据不需要另行下载（也不需要使用深度学习框架中相应的下载工具。为了实现此目标，我们对部分框架中的数据载入代码做了相应定制修改）。



### 模型checkpoint保存和载入：

建议训练过程中每个epoch结束后，保存模型训练得到checkpoint，在任务发运行发生异常时可以恢复。

本例程中（tensorflow, pytorch和mxnet不同框架），为了减少模型文件对空间的占用，在每个epoch结束后，程序仅保存了最新的checkpoint（会覆盖旧的），同时检查当前的模型对验证集的准确率，（覆盖）保留一个在验证集上表现最好的checkpoint。程序在开始时，模型会尝试载入之前最新的/最好的checkpoint，继续训练。



### 使用方法：

使用文件中心上传相应代码到`/userhome`下，提交任务json文件。相应的任务json文件如下：



#### tensorflow：

```json
{
    "jobName": "mnist-tensorflow",
    "image": "10.11.3.8:5000/pai-images/deepo:v2.0",
    "gpuType": "1080ti",
    "killAllOnCompletedTaskNumber": 1,
    "retryCount": 0,
    "taskRoles": [
        {
            "name": "mnist",
            "taskNumber": 1,
            "cpuNumber": 1,
            "memoryMB": 2048,
            "gpuNumber": 1,
            "command": "cd /userhome; python mnist-tensorflow.py"
        }
    ]
}
```



#### pytorch：

```json
{
    "jobName": "mnist-pytorch",
    "image": "10.11.3.8:5000/pai-images/deepo:v2.0",
    "gpuType": "gtx1080ti",
    "retryCount": 1,
    "taskRoles": [
        {
            "name": "mnist",
            "taskNumber": 1,
            "cpuNumber": 1,
            "memoryMB": 2048,
            "gpuNumber": 1,
            "command": "cd /userhome; python mnist-pytorch.py"
        }
    ]
}
```



#### pytorch-ignite：

```json
{
    "jobName": "mnist-pytorch-ignite",
    "image": "10.11.3.8:5000/pai-images/deepo:v2.0",
    "gpuType": "gtx1080ti",
    "retryCount": 1,
    "taskRoles": [
        {
            "name": "mnist",
            "taskNumber": 1,
            "cpuNumber": 1,
            "memoryMB": 2048,
            "gpuNumber": 1,
            "command": "pip install pytorch-ignite; cd /userhome; python mnist-pytorch-ignite.py"
        }
    ]
}
```

#### mxnet：

```json
{
    "jobName": "mnist-mxnet",
    "image": "10.11.3.8:5000/pai-images/deepo:v2.0",
    "gpuType": "gtx1080ti",
    "retryCount": 1,
    "taskRoles": [
        {
            "name": "mnist",
            "taskNumber": 1,
            "cpuNumber": 1,
            "memoryMB": 2048,
            "gpuNumber": 1,
            "command": "cd /userhome; python mnist-mxnet.py"
        }
    ]
}
```



### License:

This project is licensed under MIT License - see the LICENSE file for details.





