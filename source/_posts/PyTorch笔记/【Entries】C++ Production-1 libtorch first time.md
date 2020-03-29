---
title: 
date: 2020-02-28
tags:
categories: ["PyTorch笔记"]
mathjax: true
---

&emsp; PyTorch的主要接口为Python。虽然Python有动态编程和易于迭代的优势，但在很多情况下[1]，正是Python的这些属性会带来不利。我们经常遇到的生产环境，要满足低延迟和严格部署要求。对于生产场景而言，C++通常是首选语言，也能很方便的将其绑定到另一种语言，如Java，Rust或Go。本文主要介绍将PyTorch Python训练好的模型移植到C++中调用。
<!-- more -->

# Step1：将PyTorch模型转换为Torch Script
&emsp; PyTorch模型从Python到C++的转换由Torch Script实现。Torch Script是PyTorch模型的一种表示，可由Torch Script编译器理解、编译和序列化。如果使用基础的“eager”API编写的PyTorch模型，则必须先将模型转换为Torch Script，当然这也是比较容易的[1]。

&emsp; 将PyTorch模型转换为Torch Script有两种方法。第一种方法是Tracing，该方法通过将样本输入到模型中一次来对该过程进行评估从而捕获模型结构，并记录该样本在模型中的flow。该方法适用于模型中很少使用控制flow的模型。第二个方法就是向模型添加显式注释(Annotation)，通知Torch Script编译器它可以直接解析和编译模型代码，受Torch Script语言强加的约束。

> 小贴士 可以在官方的 [Torch Script](https://pytorch.org/docs/master/jit.html) 参考中找到这两种方法的完整文档，以及有关使用哪个方法的细节指导。


## 利用Tracing将模型转换为Torch Script
&emsp; 要通过tracing来将PyTorch模型转换为Torch脚本, 必须将模型的实例以及样本输入传递给torch.jit.trace函数。这将生成一个 torch.jit.ScriptModule 对象，并在模块的forward方法中嵌入模型评估的跟踪：


....


## 通过Annotation将Model转换为Torch Script
&emsp; 在某些情况下，例如，如果模型使用特定形式的控制流，如果想要直接在 Torch Script中编写模型并相应地标注(annotate)模型。例如，假设有以下普通的 Pytorch 模型：
```python
import torch

class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        if input.sum() > 0:
          output = self.weight.mv(input)
        else:
          output = self.weight + input
        return output
```

&emsp; 由于此模块的forward方法使用依赖于输入的控制流，因此它不适合利用 Tracing 的方法生成 Torch Script。为此,可以通过继承 torch.jit.ScriptModule 并将 @ torch.jit.script_method 标注添加到模型的 forward 中的方法，来将 model 转换为 ScriptModule：
```python
import torch

class MyModule(torch.jit.ScriptModule):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    @torch.jit.script_method
    def forward(self, input):
        if input.sum() > 0:
          output = self.weight.mv(input)
        else:
          output = self.weight + input
        return output

my_script_module = MyModule()
```


# Step2: 将Script Module序列化为一个文件
&emsp; 不论是从上面两种方法的哪一种方法获得了 ScriptModule, 都可以将得到的ScriptModule 序列化为一个文件, 然后 C++ 就可以不依赖任何 Python 代码来执行该 Script 所对应的 Pytorch 模型。假设我们想要序列化前面 trace 示例中显示的 ResNet18 模型。要执行此序列化，只需在模块上调用 save 并给个文件名：

```python
traced_script_module.save("model.pt")
```

这将在工作目录中生成一个model.pt文件。现在可以离开Python，并准备跨越到C ++语言调用。


# Step3: 在C++中加载你的 Script Module
&emsp; 要在 C++ 中加载序列化的 PyTorch 模型，应用程序必须依赖于 PyTorch C ++ API(也称为LibTorch)。LibTorch 发行版包含一组共享库，头文件和 CMake 构建配置文件。虽然 CMake 不是依赖 LibTorch 的要求，但它是推荐的方法，并且将来会得到很好的支持。这里我们将使用 CMake 和 LibTorch 构建一个最小的 C++ 应用程序，加载并执行序列化的 PyTorch 模型。

## Step3.1: 获取LibTorch
下载 [LibTorch 发行版](https://pytorch.org/)，从PyTorch网站的下载页面获取最新的稳定版本。

<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/envtool/pytorch_in_production1.jpg" width = 60% height = 60% />
</div>

下载 LibTorch 并解压缩到某个目录下，则有以下目录结构：
```bash
libtorch/
  bin/
  include/
  lib/
  share/
  build-hash
  build-version
```
lib/ 包含含链接的共享库,\
include/ 包含程序需要include的头文件,\
share/包含必要的CMake配置文件使得 find_package(Torch) \

> 小贴士: 在Windows平台上, debug and release builds are not ABI-compatible. 如果要使用debug, 要使用源码编译 PyTorch 方法。


## Step3.2: 构建应用程序
```bash
xj@ubuntu:~/dev/$ mkdir cppml
xj@ubuntu:~/dev/$ cd cppml
xj@ubuntu:~/dev/cppml$ 新建下面两个文件
CMakeLists.txt  example.cpp
```

example.cpp
```cpp
#include <iostream>
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>


int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        // return -1;
    }
    int device_id = 0;
    const char *model_path = "/home/to/dev/proml/maskface/modeltest.pt";

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA :torch::kCPU, device_id);
    torch::jit::script::Module module;
    //gpu optimize
    torch::NoGradGuard no_grad;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(model_path);

        //gpu optimize
        module.eval();
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "ok\n";


// Create a vector of inputs.
    at::Tensor input_tensor=torch::ones({1, 3, 224, 224});
    input_tensor=input_tensor.to(device);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

// Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}
```


CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(example example.cpp)
target_link_libraries(example "${TORCH_LIBRARIES}")
set_property(TARGET example PROPERTY CXX_STANDARD 11)
```

## Step3.3: 编译
```bash
xj@ubuntu:~/dev/cppml$ mkdir build
xj@ubuntu:~/dev/cppml$ cd build
xj@ubuntu:~/dev/cppml$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
xj@ubuntu:~/dev/cppml$ make
```
其中 /path/to/libtorch 是解压缩的 LibTorch 发行版的完整路径。


## Step3.4: 执行测试
```bash
xj@ubuntu:~/dev/cppml$ ./example /path/to/model.pt
```


# 在CLion IDE中构建libtorch应用程序
&emsp; 实际开发中用IDE会方便代码开发、调试，本节使用CLion构建libtorch应用程序，其实和上面的步骤差不多，只不过要稍微改下 CMakeLists.txt。 [2]

## 新建CLion工程
安装好并启动 CLion 后，新建工程，例如新建一个名为test的工程，新建好的工程目录结构如下：
<div align=center>
  <img src="https://github.com/JuneXia/JuneXia.github.io/raw/hexo/source/images/envtool/pytorch_in_production2.jpg" width = 60% height = 60% />
</div>


## 修改CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.13)
project(test)

set(CMAKE_CXX_STANDARD 11)

set(Torch_DIR /home/to/program/libtorch/share/cmake/Torch)   # 根据自己保存的路径输入

find_package(Torch REQUIRED)   # 查找库

if(NOT Torch_FOUND)
    message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)

add_executable(test main.cpp)

target_link_libraries(test "${TORCH_LIBRARIES}")   # 添加链接文件
```

编写好CMakeLists.txt后，在CLion中直接构建即可。



# 参考文献
[1] [在C++中加载PYTORCH模型](https://pytorch.apachecn.org/docs/1.0/cpp_export.html)
[2] [libtorch 的配置以及简单使用](https://jackan.cn/2018/12/23/libtorch-test/)






---------------------------------------------------------------



LibTorch各版本下载链接：\
[LibTorch library package](https://opam.ocaml.org/packages/libtorch/libtorch.1.1.0/)

[Libtorch踩坑实录：non-scalar type， '->' has non-pointer type，opencv，Expected Tensor but got Tuple](https://www.jianshu.com/p/186bcdfe9492)

[使用libtorch进行c++工程化流程](https://blog.csdn.net/qq_39016917/article/details/102976965)

[INSTALLING C++ DISTRIBUTIONS OF PYTORCH](https://pytorch.org/cppdocs/installing.html)

[PyTorch Resources](https://pytorch.org/resources/)

[torch.onnx](https://pytorch.apachecn.org/docs/0.3/onnx.html)



[TORCHSCRIPT](https://pytorch.org/docs/master/jit.html)
