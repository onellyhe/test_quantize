#ifndef DETECTOR_HPP
#define DETECTOR_HPP
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>

using namespace caffe;  // NOLINT(build/namespaces)

//定义检测器
class Detector {
 public:
  Detector(const string& model_file,
           const string& weights_file,
           const string& mean_file,
           const string& mean_value);

  std::vector<vector<float> > Detect(const cv::Mat& img);

 private:
  void SetMean(const string& mean_file, const string& mean_value);//对mean_进行初始化

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);//将 input_channels与网络输入绑定

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);//一系列的预处理

 private:
  shared_ptr<Net<float> > net_;//网络指针
  cv::Size input_geometry_; //输入图片的size
  int num_channels_; //输入图片的channel
  cv::Mat mean_; //均值图片
};


#endif // DETECTOR_HPP
