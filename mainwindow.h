#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QFileDialog>
#include <nvml.h>
#include <QTimer>
#include <detector.h>
using namespace caffe;

namespace Ui {
class MainWindow;
}

class mAP_cal
{
public:
    //shared_ptr<Net<float> > net_;
    Net<float> *net_;

    float  map_zhi;
    Ui::MainWindow * ui;

private slots:
    void on_pushButton_2_clicked();
};

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

public slots:
    int GetGPUInfo();
};

#endif // MAINWINDOW_H
