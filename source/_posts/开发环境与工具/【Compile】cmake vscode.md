---
title: 
date: 2020-04-29
tags:
categories: ["开发环境与工具"]
mathjax: true
---

VSCode + CMake 环境配置

<!-- more -->

# 直接使用 CMake 命令编译

新建以下 文件夹 和 文件：
```bash
mkdir vscode-hello && cd vscode-hello
mkdir build && touch CMakeLists.txt main.cpp

vscode-hello
├── build           # cmake 编译生成的文件都将存于该目录
├── CMakeLists.txt  # cmake编译文件
└── main.cpp        # 源代码
```

**vscode-hello/main.cpp** 内容如下:
```cpp
#include <iostream>
using namespace std;

int main()
{
    cout << "hello vscode" << endl;
    return 0;
}
```

**vscode-hello/CMakeLists.txt** 内容如下:
```cmake
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(vscode_hello)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")  # 在编译时告诉编译器产生与位置无关的代码(Position-Independent Code, 简称PIC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

SET(CMAKE_BUILD_TYPE "Debug")
# SET(CMAKE_BUILD_TYPE "Release")

add_executable(hello_exe main.cpp)
```

开始编译并执行测试：
```bash
cd /path/to/vscode-hello      # cd 到 vscode-hello 主目录下
cd build && cmake .. && make  # 编译并生成可执行文件
./hello_exe                   # 执行该可执行文件
```

<br>

实际开发中常常会用到 OpenCV 这些第三方库，在代码中增加 OpenCV 的使用，并改进 CMakeLists.txt 后如下：

增加 opencv 的 main.cpp
```cpp
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
using namespace std;

int main()
{
    cout << "hello vscode" << endl;

    char *image_path = "/home/xiajun/res/face/maskface/test-images/test_00000408.jpg";
    cv::Mat image = cv::imread(image_path, 1);
    cv::imshow("show", image);
    cv::waitKey();
    return 0;
}
```

增加 opencv 的 CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(vscode_hello)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")  # 在编译时告诉编译器产生与位置无关的代码(Position-Independent Code, 简称PIC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

SET(CMAKE_BUILD_TYPE "Debug")
# SET(CMAKE_BUILD_TYPE "Release")

set(OpenCV_DIR /home/xiajun/program/opencv-4.2.0/static_install/lib/cmake/opencv4)  # 您事先编译好的 opencv 目录

find_package(OpenCV REQUIRED)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV Not Found!")
endif(NOT OpenCV_FOUND)

include_directories(
                    ${OpenCV_INCLUDE_DIRS}
                   )

add_executable(hello_exe main.cpp)

target_link_libraries(hello_exe ${OpenCV_LIBS})
# set_property(TARGET hello_exe PROPERTY CXX_STANDARD 11)
```
编译后直接执行即可。

<br>

# 在 VSCode 中使用 CMake 编译

使用 vscode 打开刚刚新建的文件夹 vscode-hello

**step1: 配置 C++ 工程属性文件 c_cpp_properites.json** \
这一步是设置 vscode 中 C++ 工程的各种属性(比如头文件寻找路径，预定义等等)，这里主要参考了 vsvode 的官方文档[2]。这里我们可以采用通过UI界面或者通过直接修改json文件的方法来配置。[1]

按 F1 或者 ctrl+shift+p(mac用户是cmd+shift+p) 打开命令面板，并输入 edit 并选择 C/C++:Edit Configurations(UI) 或 C/C++:Edit Configurations(JSON)

vscode 会打开 C/C++ Configurations 这个配置界面，主要修改如下配置：

- Compiler Path 使用 g++
- IntelliSense Mode 使用 gcc-x64
- C 标准使用 c11 (其实默认应该就是c11)
- C++标准使用 c++11

关闭该界面，会在主目录 vscode-hello 的 .vscode 目录下生成 c_cpp_properites.json 文件：
```json
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**"
            ],
            "defines": [],
            "compilerPath": "/usr/bin/g++",
            "cStandard": "c11",
            "cppStandard": "c++11",
            "intelliSenseMode": "gcc-x64"
        }
    ],
    "version": 4
}
```

<br>

**step2: 配置编译目标文件 task.json** \
这个文件主要是告诉 vscode 你摁下生成 (ctrl+shift+b / cmd+shift+b) 的时候应该干嘛。首先我们选中并打开刚刚创建的 main.cpp 文件，这里一定要选中不然 vscode 无法知道你要对 cpp 生成。然后我们还是打开命令面板 (F1 / ctrl+shift+p / cmd+shift+p) 输入 task，并选择 Tasks: Configure Default Build Task。

然后选择 C/C++:g++ build activate file，vscode 会自动在 .vscode 文件夹下生成使用 g++ 编译文件的 tasks.json 文件。

如果只用 g++ 的话无需对文件进行修改，因为我们使用cmake，所以对文件进一步修改：
- 其中 lable 可以修改为任意的名字，这里修改 label 为 cmake build，方便我们后面的 launch.json 文件调用
- command 是调用的命令，args 是调用时紧跟在命令后的参数，修改 command 和 args，使它们组成 `cmake .. && make -j`
- group 中 isDefault 设为 true 表明为默认的生成任务
- 设置命令执行的工作目录：修改 cwd 为 "${workspaceFolder}/build"

最后修改完的 task.json 内容如下：
```cpp
{
    // See https://go.microsoft.com/fwlink/?LinkId=733558 
    // for the documentation about the tasks.json format
    "version": "2.0.0",
    "tasks": [
        {
            "type": "shell",
            "label": "cmake build",
            "command": "cmake",
            "args": [
                "..",
                "&&",
                "make",
                "-j"
            ],
            "options": {
                "cwd": "${workspaceFolder}/build"
            },
            "problemMatcher": [
                "$gcc"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            }
        }
    ]
}
```

这一步我们已经完成了生成任务的调试，如果不需要调试的话我们摁下 F5 就会在 build 目录下调用 cmake 命令进行生产，但由于缺少 launch.json 启动文件，所以应该不会成功执行。下面我们继续说怎么编写launch.json文件。

<br>

**step3: 配置启动文件 launch.json** \
同样打开命令面板输入 debug 然后选择 DEBUG:Open launch.json：然后选择C++(GDB/LLDB)，vscode 会自动在 .vscode 文件夹下生成 launch.json 文件。

- 修改 program 为 cmake 生成的文件 helle_exe (注意要和CMakeLists.txt中生成的可执行文件名一致)
- 添加 preLaunchTask 配置 来告诉 vscode 在启动调试前 要调用生成任务（如果不添加 preLaunchTask 配置的话，每次我们摁 F5 调试时不会自动生成最新的程序，需要手动生成）
- 添加 miDebuggerPath 配置 来告诉 vscode 调试器所在位置

修改后的 launch.json 完整内容如下：
```cpp
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "(gdb) 启动",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build/hello_exe",
            "args": [],
            "stopAtEntry": true,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "为 gdb 启用整齐打印",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "cmake build",
            "miDebuggerPath": "/usr/bin/gdb"
        }
    ]
}
```





https://blog.csdn.net/Ha_ku/article/details/102625837

https://www.cnblogs.com/flyinggod/p/10867530.html


https://www.hahack.com/codes/cmake/


[1] [VSCODE+CMAKE+Libtorch环境配置，实现一键编译一键DEBUG](https://blog.csdn.net/Ha_ku/article/details/102625837)
[2] [C/C++ for Visual Studio Code](https://code.visualstudio.com/docs/languages/cpp)