# 介绍  
LeinaoPAI支持分布式训练任务，每个container对外通信端口为3个，在container内部映射为10001,10002,10003，container外访问需要使用外部端口号。外部端口号随机分配，可查看环境变量`PAI_PORT_LIST_{taskRole.name}_{taskIndex}`。  
  
分布式tensorflow代码示例[mnist_replica.py](https://github.com/feng257/LeinaoPAI/blob/master/tensorflow-distributed/mnist_replica.py)  
  
编写任务提交脚本

```json
{
  "jobName": "tensorflow-distributed",
  "image": "10.11.3.8:5000/pai-images/pai.run.deepo:v1.1",
  
  "taskRoles": [
    {
      "name": "ps",
      "taskNumber": 2,
      "cpuNumber": 2,
      "memoryMB": 8192,
      "gpuNumber": 0,
      "minSucceededTaskCount": 2,
      "minFailedTaskCount": 1,      
      "command": "python /gdata/tensorflow-distributed/code/mnist_replica.py  --num_gpus=0 --batch_size=32 --data_dir=/gdata/tensorflow-distributed/data  --train_dir=/userhome/tensorflow-distributed/output --ps_hosts=$PAI_TASK_ROLE_ps_HOST_LIST --worker_hosts=$PAI_TASK_ROLE_worker_HOST_LIST --job_name=ps --task_index=$PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX"
    },
    {
      "name": "worker",
      "taskNumber": 2,
      "cpuNumber": 2,
      "memoryMB": 16384,
      "gpuNumber": 2,
      "minSucceededTaskCount": 2,(**该参数请保持与worker的taskNumber一致**)
      "minFailedTaskCount": 1,      
      "command": "python /gdata/tensorflow-distributed/code/mnist_replica.py  --num_gpus=2 --batch_size=32 --data_dir=/gdata/tensorflow-distributed/data  --train_dir=/userhome/tensorflow-distributed/output --ps_hosts=$PAI_TASK_ROLE_ps_HOST_LIST --worker_hosts=$PAI_TASK_ROLE_worker_HOST_LIST --job_name=worker --task_index=$PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX"
    }
  ],
  "retryCount": 0
}
```

可能用到的环境变量含义

| 环境变量                           | 意义                                     |
| :--------------------------------- | :--------------------------------------- |
| PAI_CURRENT_TASK_ROLE_NAME         | 任务的`taskRole.name`     |
| PAI_CURRENT_TASK_ROLE_TASK_COUNT   | 任务的`taskRole.taskNumber`  |
| PAI_TASK_ROLE_ps_HOST_LIST | `taskRole.name`是ps主机：IP列表 |
| PAI_TASK_ROLE_worker_HOST_LIST     | `taskRole.name`是worker主机：IP列表       |
| PAI_PORT_LIST_ps_0 | `taskRole.name`是ps的0号任务的IP及端口格式为`ip,port1,port2,port3` |
| PAI_PORT_LIST_worker_0 | `taskRole.name`是worker的0号任务的IP及端口格式为`ip,port1,port2,port3` |
| PAI_CURRENT_CONTAINER_IP | 当前容器的IP地址|
| PAI_CURRENT_TASK_ROLE_NAME | 当前容器的角色名称 |
| PAI_TASK_INDEX  |  当前容器的角色任务编号 |
| PAI_PORT_MAP_10001 | 容器内部映射端口10001 |
| PAI_PORT_MAP_10002 | 容器内部映射端口10002 |
| PAI_PORT_MAP_10003 | 容器内部映射端口10003 |
  
**更多环境变量信息请参考任务日志，进入任务列表-->点击具体任务-->点击 Go to Tracking Page**  
  
任务提交方式请查看[这里](https://www.bitahub.com/views/article-detail.html?articleId=_f0bf8a2c89b94945bb95c83e97815039)  
查看代码结果及模型，在[类脑测试平台](https://202.38.95.226:7443)提交测试任务，登录分配的container后，查看/userhome/tensorflow-distributed/output，具体方式请参考[这里](https://www.bitahub.com/views/article-detail.html?articleId=_f0bf8a2c89b94945bb95c83e97815039)  
