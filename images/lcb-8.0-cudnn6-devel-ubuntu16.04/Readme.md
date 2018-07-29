#### dockerfile编写示例

```
FROM ubuntu:16.04

MAINTAINER xwzheng@leinao.ai

RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \ # 修改为科大源，非必选项
    apt update && \
    apt install -y openssh-client openssh-server && \ # 为了使镜像支持ssh，需安装ssh服务
    apt clean

```

#### 镜像编译命令

```
docker build ./images/ubuntu.16.04-ssh/ -t ubuntu:16.04-ssh
```

