---
title: 
date: 2020-03-21
tags:
categories: ["PyTorch笔记"]
mathjax: true
---
本节主要讲述使用libtorch+opencv在ubuntu c++环境下sdk封装的完整示例。
<!-- more -->

# 在 CLion IDE 中使用libtorch和opencv
由于OpenCV已经安装过，所以这里暂时就不讲Opencv编译和安装的事了，这里之间使用安装好的Opencv.

其实这里要稍微改下`前文`中的 CMakeLists.txt 即可。

> 抱歉：因这是个人随笔，文章命名会随内容而动态改变，故这里引用 `前文` 来代指前面写过的系列文章。

## CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.13)
project(test)

set(CMAKE_CXX_STANDARD 11)

set(Torch_DIR /home/to/program/libtorch/share/cmake/Torch)   # 根据自己保存的路径输入
#set(OpenCV_DIR /home/to/program/opencv-3.4.0/build)   # 我的opencv之前已经安装过，并且已安装到/usr/local/lib系统路径下

find_package(Torch REQUIRED)   # 查找库
find_package(OpenCV REQUIRED)

if(NOT Torch_FOUND)
    message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)

message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(test main.cpp)

target_link_libraries(test ${TORCH_LIBRARIES} ${OpenCV_LIBS})   # 添加链接文件
```


## maskface.cpp
```cpp
//
// Created by xiajun on 20-3-17.
//

#include "maskface.h"


maskface::maskface(const std::string& filename)
{
    _margin = 0;
    _imshape[0] = 160;
    _imshape[1] = 160;
    _imshape[2] = 3;
    _module = torch::jit::load(filename);

    torch::Tensor inputs = torch::zeros({1, _imshape[2], _imshape[0], _imshape[1]});
    inputs = inputs.toType(torch::kFloat);
    _module.forward({inputs}).toTensor();
}


std::vector<std::map<int, float>> maskface::predict(cv::Mat images, std::vector<std::vector<int>> bboxes)
{
    if(images.empty())
    {
        std::cout << "ERROR: images empty, please check your image!" << std::endl;
        throw images;
    }
    if(bboxes.size() == 0 or bboxes[0].size() != 4)
    {
        std::cout << "ERROR: bboxes empty, please check your bboxes!" << std::endl;
        throw bboxes;
    }

    std::vector<std::map<int, float>> ret;
    try
    {
        std::vector<std::vector<int>>::iterator iter1;
        std::vector<int> vec;
        std::vector<cv::Mat> rois;

        // cv::cvtColor(images, images, cv::COLOR_BGR2RGB);

        auto batch_size = bboxes.size();

        for(iter1 = bboxes.begin(); iter1 != bboxes.end(); iter1++)
        {
            cv::Mat roi;
            vec = *iter1;
            //std::cout << vec << std::endl;

            vec[0] = std::max(vec[0] - _margin/2, 0);
            vec[1] = std::max(vec[1] - _margin/2, 0);
            vec[2] = std::max(vec[2] + _margin, 0);
            vec[3] = std::max(vec[3] + _margin, 0);

            //std::cout << vec << std::endl;
            cv::Rect rect(vec[0], vec[1], vec[2], vec[3]);
            cv::resize(images(rect), roi, cv::Size(_imshape[0], _imshape[1]), 0, 0, cv::INTER_CUBIC);

            rois.push_back(roi);
        }

        torch::Tensor inputs = torch::zeros({long(batch_size), _imshape[2], _imshape[0], _imshape[1]});
        inputs = inputs.toType(torch::kFloat);
        for(int i = 0; i < rois.size(); i++)
        {
//        cv::imshow("show", rois[i]);
//        cv::waitKey();

            torch::Tensor tensor_image = torch::from_blob(rois[i].data, {1, rois[i].rows, rois[i].cols, _imshape[2]}, torch::kByte);

            tensor_image = tensor_image.permute({0, 3, 1, 2});
            tensor_image = tensor_image.toType(torch::kFloat);
            tensor_image = tensor_image.div(255);
            //tensor_image = tensor_image.to(torch::kCUDA);

            tensor_image[0][0] = tensor_image[0][0].sub_(0.485).div_(0.229);
            tensor_image[0][1] = tensor_image[0][1].sub_(0.456).div_(0.224);
            tensor_image[0][2] = tensor_image[0][2].sub_(0.406).div_(0.225);
            //auto img_var = torch::autograd::make_variable(tensor_image, false);

            inputs[i] = tensor_image[0];
        }

        auto t = (double) cv::getTickCount();
        torch::Tensor result = _module.forward({inputs}).toTensor();
        at::Tensor probs = torch::softmax(result, 1);
        at::Tensor preds = torch::argmax(result, 1);
        t = (double) cv::getTickCount() - t;
        // printf("execution time = %gs\n", t / cv::getTickFrequency());


        for(int i = 0; i < batch_size; i++)
        {
            auto pred = preds[i].item<int>();
            auto prob = probs[i][pred].item<float>();

            std::map<int, float> rslt;
            rslt[pred] = prob;
            ret.push_back(rslt);
        }

        std::cout << "predict: " << ret << "\t"
                  << "execute time: " << t / cv::getTickFrequency()
                  << std::endl;


//    for(int i = 0; i < ret.size(); i++)
//    {
//        std::map<int, float> rslt = ret[i];
//        std::cout << rslt << std::endl;
//    }
    }
    catch (...)
    {
        std::cout << "ERROR: predict runtime error!" << std::endl;
    }

    return ret;
}

maskface::~maskface()
{

}
```


## maskface.h
```cpp
//
// Created by xiajun on 20-3-17.
//

#ifndef CPPML_MASKFACE_H
#define CPPML_MASKFACE_H

#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>  // dispensible on ubuntu, but must not on windows.
#include <opencv2/opencv.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/core.hpp>

class maskface {
private:
	int _margin;
	int _imshape[3];
	torch::jit::script::Module _module;

public:
	/* function: create function
	 * params: 
	 *     filename: model path */
	maskface(std::string& filename);

	/*function: predict function
	*  params:
	*      images: rgb image.
	*      bboxes: [b, 4], b denote multi-face from one image, 4 value (x_left, y_left, width, height) for every bbox respectively.
	*  return:
	*      if param_error:
	*          throw
	*      elif runtime_error:
	*          return empty vector.
	*      else:
	*          predict result, [ {label: confidence}, ... ]*/
	std::vector<std::map<int, float>> predict(cv::Mat images, std::vector<std::vector<int>> bboxes);

	~maskface();
};


#endif //CPPML_MASKFACE_H
```


## example.cpp
```cpp
//
// Created by xiajun on 20-3-17.
//

#include "maskface.h"
#include <memory>
#include <string>
#include <vector>


int main(int argc, const char *argv[]) {
    if (argc < 4) {
        std::cerr << "usage: CppProject <path-to-exported-script-module> "
                  << "<path-to-image>  <path-to-category-text>\n";
        // return -1;
    }
    const char *model_path = "/home/xiajun/dev/proml/maskface/modeltest.pt";
    const char *image_path = "/home/xiajun/res/face/maskface/test-images/test_00000008.jpg";

    maskface module = maskface(model_path);

//    cv::namedWindow("show", cv::WINDOW_AUTOSIZE);
    cv::Mat input;
    cv::Mat image = cv::imread(image_path, 1);

    std::vector<std::vector<int>> bboxes(2);
    bboxes[0].resize(4);
    bboxes[1].resize(4);
    bboxes[0][0] = 30;
    bboxes[0][1] = 50;
    bboxes[0][2] = 70;
    bboxes[0][3] = 70;

    bboxes[1][0] = 70;
    bboxes[1][1] = 10;
    bboxes[1][2] = 70;
    bboxes[1][3] = 70;

    std::vector<std::vector<int>> bboxes2;

    try
    {
        auto result = module.predict(image, bboxes);
        std::cout << result << std::endl;
    }
    catch (...)
    {
        std::cout << "catch" << std::endl;
    }

    return 0;
}
```

代码编写完成之后，直接编译构建即可。


# 参考文献
[1] `前文`

> 抱歉：因这是个人随笔，文章命名会随内容而动态改变，故这里引用 `前文` 来代指前面写过的系列文章。