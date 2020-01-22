---
title: 【开发环境与工具】Ubuntu16.04+RTX750+CUDA10.0
date: 2018-08-25
tags:
categories: ["开发环境与工具"]
mathjax: true
---
<!-- more -->

## 0. 基本环境配置
1. 检查gcc版本
```bash
gcc --version
```
对于Ubuntu 16.04来说，gcc版本需要>5.4.0。

如果没有gcc命令，则需要安装gcc
```bash
sudo apt update  # 一定要记得先update，不然找不到gcc
sudo apt install gcc
```

2. 检查Kernel版本
```bash
uname -r
```
对于Ubuntu 16.04来说，内核版本需要>4.4.0

然后需要安装对应版本的Kernel Header：
```bash
sudo apt-get install linux-headers-$(uname -r)  
```

3. 安装对应的库
```bash
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev \
libboost-all-dev libhdf5-serial-dev libgflags-dev libgoogle-glog-dev liblmdb-dev \
protobuf-compiler g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev \
libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev
```


## 1. 卸载旧的显卡驱动
```bash
# for case1: original driver installed by apt-get:
sudo apt-get remove --purge nvidia*
# 亲测上述命令执行不成功，改如下面这条命令便ok了：
sudo apt-get remove nvidia*

# for case2: original driver installed by runfile:
sudo chmod +x *.run
sudo ./NVIDIA-Linux-x86_64-384.59.run --uninstall
```

 NVIDIA Software Installer for Unix/Linux















 
  If you plan to no longer use the NVIDIA driver, you should make sure that no X screens are        
  configured to use the NVIDIA X driver in your X configuration file. If you used nvidia-xconfig to 
  configure X, it may have created a backup of your original configuration. Would you like to run   
  `nvidia-xconfig --restore-original-backup` to attempt restoration of the original X configuration 
  file?

                                 Yes                              No   
  WARNING: Your driver installation has been altered since it was initially installed; this may
           happen, for example, if you have since installed the NVIDIA driver through a mechanism
           other than nvidia-installer (such as your distribution's native package management       
           system).  nvidia-installer will attempt to uninstall as best it can.  Please see the
           file '/var/log/nvidia-uninstall.log' for details.                                        

                                                 OK  
  WARNING: Failed to delete some directories. See /var/log/nvidia-uninstall.log for details.        
                                                                                                    
                                                 OK       


如果原驱动是用apt-get安装的，就用第1种方法卸载。如果原驱动是用runfile安装的，就用–uninstall命令卸载。其实，用runfile安装的时候也会卸载掉之前的驱动，所以不手动卸载亦可。
> 卸载完成后，注意此时千万不能重启，重启电脑可能会导致无法进入系统。(亲测：其实重启也没事，笔者后面在禁用X-Window服务后遇到疑似“死机”现象，于是便强制关机重启后再继续其他步骤的。)


## 2. 禁用nouveau驱动
```bash
# 打开blacklist.conf文件：
sudo vim /etc/modprobe.d/blacklist.conf

# 在文本最后添加：
blacklist nouveau
options nouveau modeset=0

# 然后执行：
sudo update-initramfs -u

# 重启后，执行以下命令，如果没有屏幕输出，说明禁用nouveau成功：
lsmod | grep nouveau
```

## 3. 禁用X-Window服务
```bash
sudo service lightdm stop #这会关闭图形界面，但不用紧张
```

按Ctrl-Alt+F1进入命令行界面，输入用户名和密码登录即可。
> 小提示：在命令行输入：sudo service lightdm start ，然后按Ctrl-Alt+F7即可恢复到图形界面。


## 4. 命令行安装显卡驱动
[点此下载显卡驱动](https://www.nvidia.cn/Download/index.aspx)
![enter image description here](https://lh3.googleusercontent.com/-o6jQ09C2HfM/XYhRwKi2jeI/AAAAAAAAAO4/GcwJPgUtw4kwfM94ML3fo4BmdzdjtGzwQCLcBGAsYHQ/s0/nvidia-driver-download.png "nvidia-driver-download.png")
笔者亲测：对于我的ubuntu16.04 server系统，Language如若选Chinese(Simplified)则安装不成功。

```bash
#给驱动run文件赋予执行权限：
sudo chmod +x NVIDIA-Linux-x86_64-384.59.run
#后面的参数非常重要，不可省略：
sudo ./NVIDIA-Linux-x86_64-384.59.run -–no-x-check -no-nouveau-check -no-opengl-files
```
 - no-opengl-files：表示只安装驱动文件，不安装OpenGL文件。这个参数不可省略，否则会导致登陆界面死循环，英语一般称为”login loop”或者”stuck in login”。
 - no-x-check：表示安装驱动时不检查X服务，非必需。
 - no-nouveau-check：表示安装驱动时不检查nouveau，非必需。
 - Z, --disable-nouveau：禁用nouveau。此参数非必需，因为之前已经手动禁用了nouveau。
 - A：查看更多高级选项。


> 必选参数解释：因为NVIDIA的驱动默认会安装OpenGL，而Ubuntu的内核本身也有OpenGL、且与GUI显示息息相关，一旦NVIDIA的驱动覆写了OpenGL，在GUI需要动态链接OpenGL库的时候就引起问题。


**安装选项：**
> 1. There appears to already be a driver installed on your system (version:   
 390.42).  As part of installing this driver (version: 390.42), the existing   
 driver will be uninstalled.  Are you sure you want to continue?  
 Continue installation      Abort installation   
（选择 Coninue，如果是重装的话）  
> 2. The distribution-provided pre-install script failed!  Are you sure you want  
 to continue?   
 Continue installation      Abort installation   
（选择 Cotinue)  
> 3. Would you like to register the kernel module sources with DKMS? This will   
 allow DKMS to automatically build a new module, if you install a different   
 kernel later.  
  Yes                       No   
（这里选 No）  
>4. Install NVIDIA's 32-bit compatibility libraries?   
 Yes                       No   
（这里选 No）  
> 5. Installation of the kernel module for the NVIDIA Accelerated Graphics Driver  
 for Linux-x86_64 (version 390.42) is now complete.   
     OK


之后，按照提示安装，成功后重启即可。  
如果提示安装失败，不要急着重启电脑，重复以上步骤，多安装几次即可。

**Driver测试：**
```
nvidia-smi #若列出GPU的信息列表，表示驱动安装成功
nvidia-settings #若弹出设置对话框，亦表示驱动安装成功
```


## 5. 安装CUDA
**卸载旧的CUDA**
（也可以不卸载试试，据说显卡驱动可以向下兼容多个CUDA版本共存）
```bash
sudo /usr/local/cuda-8.0/bin/uninstall_cuda_8.0.pl

卸载之后，还有一些残留的文件夹，之前安装的是CUDA 8.0。可以一并删除：
sudo rm -rf /usr/local/cuda-8.0/
```


[点此下载CUDA](https://developer.nvidia.com/cuda-downloads)

```bash
sudo ./cuda_8.0.61_375.26_linux.run --no-opengl-libs
```
 - no-opengl-libs：表示只安装驱动文件，不安装OpenGL文件。必需参数，原因同上。注意：不是-no-opengl-files。
 - uninstall (deprecated)：用于卸载CUDA Driver（已废弃）。
 - toolkit：表示只安装CUDA Toolkit，不安装Driver和Samples。
 - help：查看更多高级选项。
之后，按照提示安装即可。我依次选择了：

> accept #同意安装
n #不安装Driver，因为已安装最新驱动
y #安装CUDA Toolkit
<Enter> #安装到默认目录
y #创建安装目录的软链接
n #不复制Samples，因为在安装目录下有/samples

```bash
导出CUDA的bin和lib路径： 
方法1：终端直接执行以下命令
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64

方法2：如果这台电脑只有自己一个用户，则可以在主目录下的.bashrc文件最后加入上述命令并保存，然后回到终端执行下面的命令即可
```bash
source .bashrc

方法3：如果电脑有多个用户使用，则方法2对其他用户无效，此时要将方法1中的命令加到 /etc/profile 文件的最后，然后执行下面的命令
source /etc/profile
```



安装及路径测试：输入nvcc -V 查看CUDA版本。

CUDA Sample测试：
#编译并测试设备 deviceQuery：
cd /usr/local/cuda-8.0/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery

#编译并测试带宽 bandwidthTest：
cd ../bandwidthTest
sudo make
./bandwidthTest

如果这两个测试的最后结果都是Result = PASS，说明CUDA安装成功啦。

### 笔者这一步没有安装补丁
sudo ./cuda_8.0.61.2_linux.run #最后安装补丁CUDA官方补丁



## 6. 安装cuDNN
[下载](https://developer.nvidia.com/cudnn)与CUDA版本匹配的cuDNN库，解压后执行：
```bash
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
```

## 7. 安装TensorFlow
如果没有pip，则需要先安装pip
```bash
sudo apt-get install python3-pip python3-dev
```

```bash
sudo pip3 install tensorflow-gpu==1.14.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 8. 其他python库的安装
```bash
sudo pip3 install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
sudo pip3 install opencv-contrib-python -i https://pypi.tuna.tsinghua.edu.cn/simple
sudo pip3 install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
```


[1] [Ubuntu 16.04安装NVIDIA驱动](https://blog.csdn.net/CosmosHua/article/details/76644029)
[2] [Ubuntu下Nvidia驱动安装](https://onlycaptain.github.io/2018/08/18/Ubuntu%E4%B8%8BNvidia%E9%A9%B1%E5%8A%A8%E5%AE%89%E8%A3%85/)
[3] [多版本CUDA和TensorFlow共存](https://bluesmilery.github.io/blogs/a687003b/)

