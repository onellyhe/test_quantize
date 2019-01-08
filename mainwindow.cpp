#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QWidget>
#include <stdio.h>
#include "detector.h"
#include "map.h"

#include <nvml.h>

#include <unistd.h>
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    timer = new QTimer();
    connect(timer,SIGNAL(timeout()),this,SLOT(GetGPUInfo()));
    timer->start(500);
}

MainWindow::~MainWindow()
{
    delete ui;
}

int Error(nvmlReturn_t result)
{
    result = nvmlShutdown();
    if (NVML_SUCCESS != result)
        printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));
    printf("Press ENTER to continue...\n");
    getchar();
    return 1;
}

int MainWindow::GetGPUInfo()
{
    unsigned int i=0;
    if(info_get_flag)   return 0;
    nvmlReturn_t result;

    // First initialize NVML library
    result = nvmlInit();//初始化NVML；
    if (NVML_SUCCESS != result)
    {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
        printf("Press ENTER to continue...\n");
        getchar();
        return 1;
    }
    nvmlDevice_t device;
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    result = nvmlDeviceGetHandleByIndex(i, &device);//获取设备；
    if (NVML_SUCCESS != result)
    {
        printf("Failed to get handle for device %i: %s\n", i, nvmlErrorString(result));
        return Error(result);
    }

    result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);//查询设备的名称；
    if (NVML_SUCCESS != result)
    {
        printf("Failed to get name of device %i: %s\n", i, nvmlErrorString(result));
        return Error(result);
    }
    this->ui->lineEdit_gname->setText(QLatin1String(name));

    unsigned int temperature = 0;
    result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);//获取显卡当前温度；
    if (NVML_SUCCESS != result)
    {
        printf("device %i NVML_TEMPERATURE_GPU Failed: %s\n", i, nvmlErrorString(result));
    }
    else
        this->ui->lineEdit_gtmpr->setText(QString::number(temperature,10));

    nvmlUtilization_t utilization;

    //Retrieves the current utilization rates for the device's major subsystems
    result = nvmlDeviceGetUtilizationRates(device, &utilization);
    if (NVML_SUCCESS != result)
    {
        printf(" device %i nvmlDeviceGetUtilizationRates Failed : %s\n", i, nvmlErrorString(result));
    }
    else
    {
        // Percent of time over the past sample period during which one or more kernels was executing on the GPU
        this->ui->lineEdit_guse->setText(QString::number(utilization.gpu,10)+"%");

        //Percent of time over the past sample period during which global (device) memory was being read or written
        this->ui->lineEdit_gmem->setText(QString::number(utilization.memory,10)+"%");
        //printf("GPU 使用率： %lld %% \n", utilization.gpu);
        //printf("显存使用率： %lld %% \n", utilization.memory);
    }

    unsigned int max_clock;
    result = nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_GRAPHICS, &max_clock);
    if (NVML_SUCCESS != result)
    {
        printf("device %i   nvmlDeviceGetMaxClockInfo Failed : %s\n", i, nvmlErrorString(result));
    }
/*
    unsigned int clock;
    result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &clock);
    if (NVML_SUCCESS != result)
    {
        printf("Failed to get NVML_CLOCK_GRAPHICS info for device %i: %s\n", i, nvmlErrorString(result));
    }
    else
    {
        printf("GRAPHICS： %6d Mhz   max clock ：%d  \n", clock, max_clock);
    }
 */


    return 0;
}

void MainWindow::on_pushButton_2_clicked()
{
    Caffe::SetDevice(0);
    Caffe::set_mode(Caffe::GPU);

    std::string model = "/home/onelly/model/test_bnmerge.prototxt";
    std::string weights = "/home/onelly/model/1109_bias_bnmerge.caffemodel";
    //std::string out_directory = "/home/onelly/model/Res18_SSD_1108/outdata";

    std::vector<std::string> layer_name;
    std::vector<float> in;
    std::vector<float> out;
    std::vector<float> param;
    std::vector<int> il_in, il_params, il_out;
    std::vector<int> conv_id;
    int temp;
    //第一步载入网络
    //std::shared_ptr< Net<float> > net_train;
    Net<float>* temp_net = new Net<float>(model, caffe::TEST);
    //std::cin>>temp;
    temp = 1;
    temp_net->CopyTrainedLayersFrom(weights);

    NetParameter netparam;
    //caffe::ReadNetParamsFromTextFileOrDie(model, &netparam);
    //netparam.mutable_state()->set_phase(caffe::TEST);
    temp_net->ToProto(&netparam,false);//为了进行网络改动，必须要将现有网络先记录到param中
    //此处暂时不使用trainnet
    //temp_net->CopyTrainedLayersFrom(netparam);
    //net_train.reset(temp_net);

    float temp_loss = 0;
    //const float result =
    temp_net->Forward(&temp_loss);

    temp_net->RangeInLayers(&layer_name,&in,&out,&param);
    int onelly_iter = 5;
    //run forward batches
    for(int i=0;i<onelly_iter;i++){
    temp_net->Forward(&temp_loss);
    temp_net->RangeInLayers(&layer_name,&in,&out,&param);
    }
    //push back il
    for (int i = 0; i < layer_name.size(); ++i) {
    il_in.push_back((int)ceil(log2(in[i])));
    il_out.push_back((int)ceil(log2(out[i])));
    il_params.push_back((int)ceil(log2(param[i])+1));
    }
    for(int i = 0; i < layer_name.size(); ++i){
      for(int j = 0; j< netparam.layer_size();++j){
          if( netparam.layer(j).name()==layer_name[i]){
              conv_id.push_back(j);
          }
      }
    }
        // for (int i = 0; i < param->layer_size(); ++i){
        //     if ( param->layer(i).type() == "Convolution"){
        //         std::cout<<param->layer(i).name()<<": "<<i<<std::endl;
        //     }
        // }
    for(int i=0;i<10;i++){
      std::cout<<"name of layer "<<i<<": "<<temp_net->layers()[i]->type()<<std::endl;
    }
    for (int k = 0; k < layer_name.size(); ++k) {
    LOG(INFO) << "Layer " << layer_name[k] <<
        ", integer length input=" << il_in[k] <<
        ", integer length output=" << il_out[k] <<
        ", integer length parameters=" << il_params[k]<<
        ", integer conv_id=" << conv_id[k];
    }


    //onelly TODO: delete net
    //delete &*net_train;
    //net_train.reset();


    //trans a single layer to ristretto
    EditConvolution2DynamicFixedPoint(&netparam, il_in, il_params, il_out, conv_id,0,8);
    //   if (Caffe::root_solver()) {
    //     net_train.reset(new Net<float>(netparam));
    //   } else {
    net_train.reset(new Net<float>(netparam));
    std::cout<<"129"<<std::endl;
    net_train->CopyTrainedLayersFrom(netparam);
    std::cout<<"131"<<std::endl;
    //   }
        //net_train = new Net<float>(netparam, NULL);
        //net_train->CopyTrainedLayersFrom(weights);

    for(int i=0;i<layer_name.size();i++){
      std::cout<< layer_name[i]<<"\t"<< in[i] << "\t"
            << out[i] << "\t" << param[i] << std::endl;
    }
    std::cout << "temp_loss: " << temp_loss << std::endl;
    for(int i=0;i<10;i++){
      std::cout<<"name of layer "<<i<<": "<<net_train->layers()[i]->type()<<std::endl;
    }
    for (int k = 0; k < layer_name.size(); ++k) {
    LOG(INFO) << "Layer " << layer_name[k] <<
        ", integer length input=" << il_in[k] <<
        ", integer length output=" << il_out[k] <<
        ", integer length parameters=" << il_params[k]<<
        ", integer conv_id=" << conv_id[k];
    }
    layer_name.clear();
    for(int i=0;i<onelly_iter;i++){
        net_train->Forward(&temp_loss);
        net_train->RangeInLayers(&layer_name,&in,&out,&param);
    }

    for(int i=0;i<layer_name.size();i++){
        std::cout<< layer_name[i]<<"\t"<< in[i] << "\t"
            << out[i] << "\t" << param[i] << std::endl;
    }



    return 0;

}
