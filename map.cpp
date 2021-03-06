#include "map.h"
#include <mainwindow.h>
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



#define CHECK_EQ(val1, val2) CHECK_OP(_EQ, ==, val1, val2)
#define CHECK_NE(val1, val2) CHECK_OP(_NE, !=, val1, val2)
#define CHECK_LE(val1, val2) CHECK_OP(_LE, <=, val1, val2)
#define CHECK_LT(val1, val2) CHECK_OP(_LT, < , val1, val2)
#define CHECK_GE(val1, val2) CHECK_OP(_GE, >=, val1, val2)
#define CHECK_GT(val1, val2) CHECK_OP(_GT, > , val1, val2)
template <typename T>
bool SortScorePairDescend(const pair<float, T>& pair1,
                          const pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

void CumSum(const vector<pair<float, int> >& pairs, vector<int>* cumsum) {
  // Sort the pairs based on first item of the pair.
  vector<pair<float, int> > sort_pairs = pairs;//这个pair 中的第一个参数是得分 第二个参数是指
  std::stable_sort(sort_pairs.begin(), sort_pairs.end(),
                   SortScorePairDescend<int>);

  cumsum->clear();
  for (int i = 0; i < sort_pairs.size(); ++i) {
    if (i == 0) {
      cumsum->push_back(sort_pairs[i].second);
    } else {
      cumsum->push_back(cumsum->back() + sort_pairs[i].second);
     // std::cout<<cumsum->back()<<std::endl;
    }
  }
}
void ComputeAP(const vector<pair<float, int> >& tp, const int num_pos,
               const vector<pair<float, int> >& fp, const string ap_version,
               vector<float>* prec, vector<float>* rec, float* ap) {
  const float eps = 1e-6;
  //std::cout<<"come in:"<<std::endl;
  CHECK_EQ(tp.size(), fp.size()) << "tp must have same size as fp.";
  const int num = tp.size();
  //Make sure that tp and fp have complement value.
  for (int i = 0; i < num; ++i) {
    CHECK_LE(fabs(tp[i].first - fp[i].first), eps);
    //std::cout<<"tp and fp"<<tp[i].first<<"  "<<tp[i].second<<" "<<fp[i].first<<" "<<fp[i].second<<std::endl;
    CHECK_EQ(tp[i].second, 1 - fp[i].second);
  }
  //经过计算可以得出 上述的tp 和fp 的计算是指 《得分，标定（指正确和错误）》
  prec->clear();
  rec->clear();
  *ap = 0;
  if (tp.size() == 0 || num_pos == 0) {
    return;
  }

  // Compute cumsum of tp.
  vector<int> tp_cumsum;
  CumSum(tp, &tp_cumsum);
  CHECK_EQ(tp_cumsum.size(), num);

  // Compute cumsum of fp.
  vector<int> fp_cumsum;
  CumSum(fp, &fp_cumsum);
  CHECK_EQ(fp_cumsum.size(), num);

  // Compute precision.
  for (int i = 0; i < num; ++i) {
    prec->push_back(static_cast<float>(tp_cumsum[i]) / (tp_cumsum[i] + fp_cumsum[i]));
   // std::cout<<tp_cumsum[i]<<"<-    ->"<<fp_cumsum[i]<<std::endl;
   // std::cout<<"label : "<<i<<"   precision :"<<static_cast<float>(tp_cumsum[i]) / (tp_cumsum[i] + fp_cumsum[i])<<std::endl;
  }

  // Compute recall.
  for (int i = 0; i < num; ++i) {
    CHECK_LE(tp_cumsum[i], num_pos);
    rec->push_back(static_cast<float>(tp_cumsum[i]) / num_pos);
  //  std::cout<<"tp_cusum: "<<tp_cumsum[i]<<"  num_pos:"<<num_pos<<std::endl;
 //   std::cout<<"labels :"<<i<<" recall:"<<static_cast<float>(tp_cumsum[i]) / num_pos<<std::endl;
  }

  if (ap_version == "11point") {
    // VOC2007 style for computing AP.
    vector<float> max_precs(11, 0.);
    int start_idx = num - 1;
    for (int j = 10; j >= 0; --j) {
      for (int i = start_idx; i >= 0 ; --i) {
        if ((*rec)[i] < j / 10.) {
          start_idx = i;
          if (j > 0) {
            max_precs[j-1] = max_precs[j];
          }
          break;
        } else {
          if (max_precs[j] < (*prec)[i]) {
            max_precs[j] = (*prec)[i];
          }
        }
      }
    }
    for (int j = 10; j >= 0; --j) {
      *ap += max_precs[j] / 11;
    }
  } else if (ap_version == "MaxIntegral") {
    // VOC2012 or ILSVRC style for computing AP.
    float cur_rec = rec->back();
    float cur_prec = prec->back();
    for (int i = num - 2; i >= 0; --i) {
      cur_prec = std::max<float>((*prec)[i], cur_prec);
      if (fabs(cur_rec - (*rec)[i]) > eps) {
        *ap += cur_prec * fabs(cur_rec - (*rec)[i]);
      }
      cur_rec = (*rec)[i];
    }
    *ap += cur_rec * cur_prec;
  } else if (ap_version == "Integral") {
    // Natural integral.
    float prev_rec = 0.;
    for (int i = 0; i < num; ++i) {
      if (fabs((*rec)[i] - prev_rec) > eps) {
        *ap += (*prec)[i] * fabs((*rec)[i] - prev_rec);
      }
      prev_rec = (*rec)[i];
    }
  } else {
    std::cout << "Unknown ap_version: " << ap_version;
  }
}

mAP_cal::mAP_cal(NetParameter netparam,std::string weights):netparam(netparam),weights(weights){
    std::cout<<"term:"<<std::endl;
}



//float mAP_calc(shared_ptr<Net<float> > net_)
void  mAP_cal::mAP_calc()
{
      //shared_ptr<Net<float> > net_;//
//      shared_ptr<Net<float> > net_ = (Net<float>*)( arg);
      //shared_ptr<Net<float> > net_ = * net1_;
      //mAP_cal * map_c =  (mAP_cal *) (arg);
      //shared_ptr<Net<float> > net_;

    net_ = new Net<float>(netparam,NULL);//从model_file中读取网络结构，初始化网络

    std::cout<<"create successful"<<std::endl;
    net_->CopyTrainedLayersFrom(netparam);
    //net_->CopyTrainedLayersFrom(weights);//从权值文件中读取网络参数，初始化net_
    std::cout<<"load weights successful"<<std::endl;
      map<int, map<int, vector<pair<float, int> > > > all_true_pos;
      map<int, map<int, vector<pair<float, int> > > > all_false_pos;
      map<int, map<int, int> > all_num_pos;
     // net_.reset(new Net<float>("/home/zhao/ssd/caffe/data/123/test1.prototxt", TEST));// /home/zhao/ssd/caffe/data/123/test1.prototxt /home/zhao/ssd/caffe/data/SSD_300x300/test.prototxt
     // net_->CopyTrainedLayersFrom("/home/zhao/ssd/caffe/data/123/_iter_500000.caffemodel");///home/zhao/ssd/caffe/data/123/_iter_500000.caffemodel /home/zhao/ssd/caffe/data/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel
      Caffe::set_mode(Caffe::GPU);
      //net_->Forward();
      vector<Blob<float>*> result = net_->output_blobs();
      std::cout<<result.size()<<std::endl;
      //const float* result = result_blob->cpu_data();
      //const int num_det = result_blob->height();
      int terms = 100;
      float mAPs=0;
      float mAP;
      for(int i= 0;i<terms;i++)
      {
          std::cout<<"term:"<<i<<std::endl;
      net_->Forward();
      for (int j = 0; j < result.size(); ++j) {
          CHECK_EQ(result[j]->width(), 5);
          //std::cout<<result[j]->width()<<std::endl;//验证得到输出的结果是5
          const float* result_vec = result[j]->cpu_data();
          int num_det = result[j]->height();//显示有几个检测的结果
          for (int k = 0; k < num_det; ++k) {
            int item_id = static_cast<int>(result_vec[k * 5]);
            int label = static_cast<int>(result_vec[k * 5 + 1]);
            if (item_id == -1) {
              // Special row of storing number of positives for a label.
              if (all_num_pos[j].find(label) == all_num_pos[j].end()) {
                all_num_pos[j][label] = static_cast<int>(result_vec[k * 5 + 2]);
              } else {
                all_num_pos[j][label] += static_cast<int>(result_vec[k * 5 + 2]);
              }
            } else {
              // Normal row storing detection status.
              float score = result_vec[k * 5 + 2];
              int tp = static_cast<int>(result_vec[k * 5 + 3]);
              int fp = static_cast<int>(result_vec[k * 5 + 4]);
              if (tp == 0 && fp == 0) {
                // Ignore such case. It happens when a detection bbox is matched to
                // a difficult gt bbox and we don't evaluate on difficult gt bbox.
                continue;
              }
              all_true_pos[j][label].push_back(std::make_pair(score, tp));//经过检测可以得到 首先all_true_pos 的第一个参数值是1 第二个参数的值是7 也就是种类数
              all_false_pos[j][label].push_back(std::make_pair(score, fp));
            }
          }
        }
      //std::cout<<"all_true_pos"<<all_true_pos[0].size()<<std::endl;
      for (int i = 0; i < all_true_pos.size(); ++i) {
        if (all_true_pos.find(i) == all_true_pos.end()) {
          std::cout << "Missing output_blob true_pos: " << i;
        }
        const map<int, vector<pair<float, int> > >& true_pos =
            all_true_pos.find(i)->second;
        if (all_false_pos.find(i) == all_false_pos.end()) {
          std::cout << "Missing output_blob false_pos: " << i;
        }
        const map<int, vector<pair<float, int> > >& false_pos =
            all_false_pos.find(i)->second;
        if (all_num_pos.find(i) == all_num_pos.end()) {
          std::cout<< "Missing output_blob num_pos: " << i;
        }
        const map<int, int>& num_pos = all_num_pos.find(i)->second;
        map<int, float> APs;
        mAP = 0.;
        // Sort true_pos and false_pos with descend scores.
        for (map<int, int>::const_iterator it = num_pos.begin();
             it != num_pos.end(); ++it) {
          int label = it->first;
          int label_num_pos = it->second;
          if (true_pos.find(label) == true_pos.end()) {
            std::cout << "Missing true_pos for label: " << label;
            continue;
          }
          const vector<pair<float, int> >& label_true_pos =
              true_pos.find(label)->second;
          if (false_pos.find(label) == false_pos.end())
          {
            std::cout << "Missing false_pos for label: " << label<<std::endl;
            continue;
          }
          const vector<pair<float, int> >& label_false_pos =
              false_pos.find(label)->second;
          vector<float> prec, rec;
          ComputeAP(label_true_pos, label_num_pos, label_false_pos,
                  "Integral", &prec, &rec, &(APs[label]));//"11point" param_.ap_version()
          mAP += APs[label];

          //std::cout<<"Test net output label # :"<<label <<"   ap is:"<<APs[label]<<std::endl;
        }
        mAP /= num_pos.size();
        std::cout<<mAP<<std::endl;
        const int output_blob_index = net_->output_blob_indices()[i];
        const string& output_name = net_->blob_names()[output_blob_index];
        std::cout << "    Test net output #" << i << ": " << output_name << " = "
                  << mAP<<std::endl;

      }
      mAPs = mAPs + mAP;
      }
      mAP = mAPs/(float)terms;
      map_zhi =  mAP;
      char c[10];
      //float mAP =  map_c.map_zhi;
      int length = sprintf(c, "%lf", mAP);
      std::cout<<mAP<<std::endl;
    //  map_c->ui->mAP->setText(c);
    //  map_c->ui->lineEdit_mAP->setText(QString::number(mAP));
     // map_c = 0;
     // std::cout<<mAP<<std::endl;
      //return ((void*)1);

//      map_c->ui->mAP->setText("mAP");
//      map_c->ui->map_label->setText(QString::number(float(int(mAP*10000))/10000));
//      map_c->ui->Detect_video->setEnabled(true);
//      map_c->ui->Detect_picture->setEnabled(true);
//      map_c->ui->mAP->setEnabled(true);
//      map_c->ui->Reset->setEnabled(true);
//      map_c->ui->Select_picture->setEnabled(true);
//      map_c->ui->Select_video->setEnabled(true);
//      map_c->ui->lineEdit_picture->setEnabled((true));
//      map_c->ui->lineEdit_video->setEnabled((true));
}



