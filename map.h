#ifndef MAP_H
#define MAP_H
#include <caffe/caffe.hpp>
#define USE_OPENCV
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <iostream>
//#include <mainwindow.h>
//#include <QMainWindow>
using namespace caffe;

class mAP_cal
{
public:
    explicit mAP_cal(NetParameter netparam,std::string weights);
    ~mAP_cal(){

    }
    //string model = "";
    NetParameter netparam;
    std::string weights = "";
    Net<float>* net_ = NULL;
    float  map_zhi = 0.0;
    //Ui::MainWindow * ui;
    void  mAP_calc();

};

template <typename T>
bool SortScorePairDescend(const pair<float, T>& pair1, const pair<float, T>& pair2);
void CumSum(const vector<pair<float, int> >& pairs, vector<int>* cumsum);
void ComputeAP(const vector<pair<float, int> >& tp, const int num_pos,
               const vector<pair<float, int> >& fp, const string ap_version,
               vector<float>* prec, vector<float>* rec, float* ap);
//float mAP_calc(shared_ptr<Net<float> > net_);

//extern shared_ptr<Net<float> > net_;
#endif // MAP_H
