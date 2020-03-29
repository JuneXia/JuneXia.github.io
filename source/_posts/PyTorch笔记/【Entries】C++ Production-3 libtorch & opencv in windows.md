---
title: 
date: 2020-03-21
tags:
categories: ["PyTorch笔记"]
mathjax: true
---
本节主要讲述在 windows c++ 下使用 libtorch + opencv 的部署，感觉比在ubuntu下坑多了，网上很多例子都不能成功执行，这里总结下我的方法，部分参考文献[1]。
<!-- more -->

> 笔者友情提示：windows有的时候会有一些莫名其妙的错误，如果您是在 ubuntu 下编写好的c++代码，然后直接拿到 windows 构建后不能正确编译或运行，则可尝试自己在windows下新建c++代码文件，然后将ubuntu下编写好的代码内容拷贝进来。

# 第一步：
下载libtorch和opencv，尽量使用较新版本的，我这里使用的libtorch是1.4版的，opencv是4.2版的。

下载完成后解压到自己的指定目录下，例如我把它们都放在了 F:/program 目录下。

添加环境变量：
```
libtorch 的 .dll 目录：F:/program/libtorch-win-shared-with-deps-1.4.0/libtorch/lib)
OpenCV 的 .dll 目录：F:/program/opencv-4.2.0-vc14_vc15/build/x64/vc14/lib)
```
将上面两个目录都添加到系统环境变量PATH中去。


# 第一步：模型转换
PyTorch的C++版本用的Torch Script，官方给了两种将pytorch模型转换成Torch Script的方法，这一步在前文已经说过了，这里不再赘述。

第三步有两种方法，分别是 “VS 构建方法” 和 “CMake 构建方法”，下面注意介绍。

# 第三步：方法一（VS构建方法）
打开Visual Studio，新建工程后，右击项目，选择`属性`，配置如下：

**step1**: “配置属性>VC++目录>包含目录”配置：
```
F:\program\libtorch-win-shared-with-deps-1.4.0\libtorch\include
F:\program\opencv-4.2.0-vc14_vc15\build\include
F:\program\opencv-4.2.0-vc14_vc15\build\include\opencv
F:\program\opencv-4.2.0-vc14_vc15\build\include\opencv2
```

**step2**: “配置属性>VC++目录>库目录”配置：
```
F:\program\libtorch-win-shared-with-deps-1.4.0\libtorch\lib
F:\program\opencv-4.2.0-vc14_vc15\build\x64\vc14\lib
```

**step3**: “配置属性>链接器>输入”配置：
```
c10.lib
caffe2_module_test_dynamic.lib
clog.lib
cpuinfo.lib
libprotobuf.lib
libprotobuf-lite.lib
libprotoc.lib
torch.lib
opencv_world420.lib
```

文献[1]说按照上面的配置完后 还有下面的两个地方需要修改：（我实测下面的两个不改也行）\
第一项：属性->C/C++ ->常规->SDL检查->否。\
第二项：属性->C/C++ ->语言->符号模式->否。

配置好环境后，就可以开始编写代码了，这里使用 《libtorch & opencv in ubuntu》中的代码。\
**注意**：“maskface.h” 中的 “#include <torch/torch.h>” 在 ubuntu 中是可有可无的，但是在 windows 亲测会出错，故在 windows 中这行代码可以直接注释掉。


# 第三步：方法二（CMake构建方法）
在windows下除了能够用vs来构建项目，也可以通过cmake来构建，笔者亲测直接使用ubuntu下编写好的CMakeLists.txt文件放在windows下并不能成功。

**step1**: 首先在自己指定的目录下新建一个文件夹（例如cppml），后面所有的代码及配置文件全部放于该文件夹下。

**step2**: 编写代码
这里继续使用前文已经编写好的 c++ 代码。


**step3**: 编写CMakeLists.txt文件
下面是在前文ubuntu下编写CMakeLists.txt的基础上并参考文献[1]后所改进后的 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cppml)

SET(CMAKE_BUILE_TYPE RELEASE)

INCLUDE_DIRECTORIES(
F:/program/libtorch-win-shared-with-deps-1.4.0/libtorch/include
F:/program/opencv-4.2.0-vc14_vc15/build/include
F:/program/opencv-4.2.0-vc14_vc15/build/include/opencv
F:/program/opencv-4.2.0-vc14_vc15/build/include/opencv2
)

SET(Torch_LIBRARIES F:/program/libtorch-win-shared-with-deps-1.4.0/libtorch/lib)
SET(OpenCV_LIBS F:/program/opencv-4.2.0-vc14_vc15/build/x64/vc14/lib)

LINK_DIRECTORIES(
${Torch_LIBRARIES}
${OpenCV_LIBS}
)

add_executable(cppml maskface.cpp maskface.h example.cpp)

# 注意对于不同版本的libtorch或opencv来说链接库文件可能不一样
target_link_libraries(cppml
c10.lib
caffe2_module_test_dynamic.lib
clog.lib
cpuinfo.lib
libprotobuf.lib
libprotobuf-lite.lib
libprotoc.lib
torch.lib
opencv_world420.lib
)

set_property(TARGET cppml PROPERTY CXX_STANDARD 11)
```


**step4**: 新建build目录
在代码文件的同级目录下，新建一个build目录，用于 cmake 编译的目标目录，
```cmake
F:/path/to/your/cppml:
.
├── build
├── CMakeLists.txt
├── example.cpp
├── maskface.cpp
└── maskface.h
```

**step5**: 使用cmake编译
打开 cmake gui 界面，配置项目源代码路径以及目标路径：
```cmake
where is the source code: F:/path/to/your/cppml  # 这里是刚刚我们建立的项目目录
where to build the binaries: F:/path/to/your/cppml/build  # 目标目录
```

点击Configure进行配置，根据自己的 Visual Studio 版本选择对应的 generator，例如我这里选择的是 “Visual Studio 14 2015 Win64” 。其他的默认即可。

待出现 Configure done 后，再点击一次 Configure，待再次 Configure done 后，再点击 Generate 生成即可。

完成上述过程就可以打开项目，在vs项目中即可编译并运行代码了。

> 注意：如果编译的时候使用的是 Release 版的 libtorch 或者 OpneCV，则在VS中也应该对应的设置为 Release 模式。反之如果使用的是 Debug 版的 libtorch 或者 OpneCV，则在VS中也应该设置为 Debug 模式。

如果运行时提示缺少 .dll 文件，则应该将 libtorch 和 OpneCV 的 bin 目录加入到系统环境变量中，也可以把 lib 文件夹下的 .dll 全部拷贝到我们的 .cpp 所在目录下。


# 参考文献
[1] [C++调用pytorch，LibTorch在win10下的vs配置和cmake的配置](https://www.zerahhah.com/article/20)
