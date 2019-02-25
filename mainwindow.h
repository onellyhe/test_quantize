#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QFileDialog>
#include <nvml.h>
#include <QTimer>
#include <detector.h>
#include <layerinfo.h>
#include <vector>
#include <QStringListModel>
#include <QListView>
#include "map.h"
using namespace caffe;

#include <QDebug>

namespace Ui {
class MainWindow;
}



class MainWindow : public QMainWindow
{
    Q_OBJECT

public:    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    bool info_get_flag = false;
    double Tflops_per_picture = 0.0289;
    double Tflops = 0;
    double TflopsRate=0;

private:
    Ui::MainWindow *ui;
    QTimer  *timer;
    std::vector<LayerInfo*> layers1;
    std::vector<LayerInfo*> layers2;
    void refreshList1();
    void refreshList2();
    QStringList strList1;
    QStringListModel *model1 = NULL;
    QStringList strList2;
    QStringListModel *model2 = NULL;
    void showList1(int current);
    void showList2(int current);
    std::string model = "/home/onelly/model/SSD_ResNet/test1_openbn.prototxt";
    std::string weights = "/home/onelly/model/SSD_ResNet/temp_voc.caffemodel";
    std::string out_directory = "/home/onelly/model/SSD_ResNet/quantized_Net";
    mAP_cal *map_cal = NULL;

public slots:
    int GetGPUInfo();
    void on_pushButton_2_clicked();
private slots:
    void on_layerList1_doubleClicked(const QModelIndex &index);
    void on_pushButton_5_clicked();
    void on_layerList2_doubleClicked(const QModelIndex &index);
    void on_Select_prototxt_clicked();
    NetParameter EditConvolution2DynamicFixedPoint();
    void on_pushButton_3_clicked();
};


//class mAP_cal
//{
//public:
//    //shared_ptr<Net<float> > net_;
//    Net<float> *net_;

//    float  map_zhi;
//    Ui::MainWindow * ui;

//};

#endif // MAINWINDOW_H
