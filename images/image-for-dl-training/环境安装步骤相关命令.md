# 环境安装步骤

## 1 ssh工具安装

选择一个ssh工具，可以是下面的某一种：

* XManager https://pan.baidu.com/s/1KCqrrDXPq0rOgpNMIriX9A  提取码：nnn8，或者http://u.163.com/swnf7p4t  提取码: PAYnm3fY
* putty
* 如果本身是linux或mac，可以直接使用ssh命令

## 2 远程连接到各自对应的linux环境

在远程连接工具中执行如下命令, 请填写自己的序号
```
ssh leinao@36.7.159.236 -p $(( 10100 + 序号))
```

**提示:**

* 默认密码为 ****1111，请及时更改
* 把 $(( 10100 + 序号)) 替换为具体数值，如第10号机器，其端口为 10110
* 如果用xshell登录工具，不要 -p

## 3 英伟达驱动安装

1） 安装驱动
```
cd /home/leinao
sudo ./NVIDIA-Linux-x86_64-390.59.run -s
```
2） 验证是否安装成功,执行如下命令
```
nvidia-smi
```

## 4 Docker安装
1) 安装命令
```
sudo apt-get update
sudo apt-get -y install apt-transport-https  ca-certificates  curl  software-properties-common
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get -y install docker-ce
```
2) 验证是否安装
```
sudo docker run hello-world
```

**提示：**

* 不要用sudo apt install -y docker.io来安装docker（这个docker版本低，不符合后续要求）
* 如果已经安装，可以用 sudo apt remove docker.io来卸载

## 5 nvidia-docker安装

1) 安装命令
```
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey |  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list |   sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
```
2) 验证命令
```
sudo docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi
```

## 6 载入镜像
执行如下命令
```
sudo docker load -i leinaotraining.tar
```

## 7 创建容器
执行如下命令
```
sudo docker run --runtime=nvidia -ti --ipc=host   -v /home/leinao/result:/result -p 10022:22 --name leinao leinao/tensorflow-mxnet-py36-cu90 bash
```
命令执行成功后,命令行交互已经处于容器内,可以在容器内进行需要的编程操作

验证命令：

```
python
import tensorflow as tf
```

**提示：**

* 如果要从容器返回宿主机，可以用 exit 命令
* 判断当前是处于宿主机还是容器：如果命令提示符上的主机名有”ubuntu“字样，处于宿主机；如果主机名是一串”数字+字母“组合，处于容器
* 运行深度学习框架，需要处于容器内

## 8 重新进入

检测容器是否处于运行状态

```
sudo docker ps
```

如果未发现名为"leinao"的容器，则运行

```
sudo docker start leinao
```

从宿主机进入到容器

```
sudo docker exec -ti leinao bash
```



## 9 可选步骤

把容器配置成可以直接ssh登录。以下操作是在容器内执行的：
```
apt update && apt install -y openssh-server
service ssh start
# 设置密码
adduser leinao 
```
这样之后可以通过 ssh leinao@36.7.159.236 -p $(( 10200 + 序号)) 直接登录容器



## 10 数据集准备

利用scp命令，从局域网主机拷贝文件到本地。以下操作是在宿主机上进行

```
sudo scp -r leinao@10.10.5.95:/home/leinao/mnist_DL /home/leinao/result/
```

当提示需要密码时，输入默认密码****1111

其中`/home/leinao/mnist_DL`放置的是`深度学习实战`课程所需的材料

宿主机上的`/home/leinao/result`目录等同于容器内的`/result`目录

