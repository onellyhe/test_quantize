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

    ui->layerList1->setEditTriggers(QAbstractItemView::NoEditTriggers);
    ui->layerList2->setEditTriggers(QAbstractItemView::NoEditTriggers);


}

MainWindow::~MainWindow()
{
    if(map_cal != NULL){
        delete map_cal;
        map_cal = NULL;
    }
    for(int i=0;i<layers1.size();i++)
        delete layers1[i];
    for(int i=0;i<layers2.size();i++)
        delete layers2[i];
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

NetParameter MainWindow::EditConvolution2DynamicFixedPoint() {

        // for (int i = 0; i < param->layer_size(); ++i){
        //     if ( param->layer(i).type() == "Convolution"){
        //         std::cout<<param->layer(i).name()<<": "<<i<<std::endl;
        //     }
        // }

    //第一步init Network载入网络
    Net<float>* temp_net = new Net<float>(model, caffe::TEST);
    int temp = 1;
    temp_net->CopyTrainedLayersFrom(weights);

    NetParameter netparam;
    temp_net->ToProto(&netparam,false);//为了进行网络改动，必须要将现有网络先记录到param中。
    for(int i=0;i<layers2.size();i++){
        LayerParameter* param_layer = netparam.mutable_layer(layers2[i]->layerID);

        param_layer->set_name(layers2[i]->layerName);
        param_layer->set_type(layers2[i]->getTypeName());
        //param
        param_layer->mutable_quantization_param()->set_fl_params(layers2[i]->bitWide -
            layers2[i]->paramIl);
        param_layer->mutable_quantization_param()->set_bw_params(layers2[i]->bitWide);
        //in
        param_layer->mutable_quantization_param()->set_fl_layer_in(layers2[i]->bitWide -
            layers2[i]->inIl);
        param_layer->mutable_quantization_param()->set_bw_layer_in(layers2[i]->bitWide);
        //out
        param_layer->mutable_quantization_param()->set_fl_layer_out(layers2[i]->bitWide -
            layers2[i]->outIl);
        param_layer->mutable_quantization_param()->set_bw_layer_out(layers2[i]->bitWide);
    }
    return netparam;
}

//========================================================================================================

void MainWindow::on_pushButton_2_clicked()
{
    strList1.clear();
    strList2.clear();
    for(int i=0;i<layers1.size();i++)
        delete layers1[i];
    for(int i=0;i<layers2.size();i++)
        delete layers2[i];
    layers1.clear();
    layers2.clear();
    ui->layerDetails1->clear();
    ui->layerDetails2->clear();
    ui->layerID1->clear();
    ui->layerName1->clear();


    Caffe::SetDevice(0);
    Caffe::set_mode(Caffe::GPU);


    std::vector<std::string> layer_name;//层名记录
    std::vector<float> in;//层输入绝对值最大值
    std::vector<float> out;//层输入绝对值最大值
    std::vector<float> param;//权值绝对值最大值
//    std::vector<int> il_in, il_params, il_out;//输入、权值、输出的il值
    std::vector<int> layer_id;//层名对应层号
    std::vector<LayerInfo::LayerType> types;//类型
    int temp;
    //第一步init Network载入网络
    //std::shared_ptr< Net<float> > net_train;
    Net<float>* temp_net = new Net<float>(model, caffe::TEST);
    //std::cin>>temp;
    temp = 1;
    temp_net->CopyTrainedLayersFrom(weights);

    NetParameter netparam;
    //caffe::ReadNetParamsFromTextFileOrDie(model, &netparam);
    //netparam.mutable_state()->set_phase(caffe::TEST);
    temp_net->ToProto(&netparam,false);//为了进行网络改动，必须要将现有网络先记录到param中。
    //此处暂时不使用trainnet
    //temp_net->CopyTrainedLayersFrom(netparam);
    //net_train.reset(temp_net);

    //第二步init Range：初始进行网络Range计算，初始确定网络结构中可以量化的层和in、out、weights的maxabs。
    float temp_loss = 0;
    //const float result =
    temp_net->Forward(&temp_loss);

    temp_net->RangeInLayers(&layer_name,&in,&out,&param);
    int onelly_iter = 5;
    //第三步run forward batches 进行多次前向传播，寻找各层最大值。
    for(int i=0;i<onelly_iter;i++){
        temp_net->Forward(&temp_loss);
        temp_net->RangeInLayers(&layer_name,&in,&out,&param);
    }
//    //第四步push back il按照顺序将il值压入vector中。
//    for (int i = 0; i < layer_name.size(); ++i) {
//        il_in.push_back((int)ceil(log2(in[i])));
//        il_out.push_back((int)ceil(log2(out[i])));
//        il_params.push_back((int)ceil(log2(param[i])+1));
//    }
    //第五步find layerID找到layer_name中对应的layerID,对应找到层的类型
    for(int i = 0; i < layer_name.size(); ++i){
      for(int j = 0; j< netparam.layer_size();++j){
          if( netparam.layer(j).name()==layer_name[i]){
              layer_id.push_back(j);
              LayerParameter* lp = netparam.mutable_layer(j);

              if(strcmp(lp->type().c_str(), "Convolution") == 0){
                  types.push_back(LayerInfo::LayerType::CONVOLUTION);
              }else if(strcmp(lp->type().c_str(), "ConvolutionRistretto") == 0){
                  types.push_back(LayerInfo::LayerType::CONV_RISTRETTO);
              }else if(strcmp(lp->type().c_str(), "InnerProduct") == 0){
                  types.push_back(LayerInfo::LayerType::FULLCONNECTION);
              }else {
                  types.push_back(LayerInfo::LayerType::FC_RISTRETTO);
              }
          }
      }
    }
    //删除网络
    delete temp_net;
    //第六步放到layerinfo类内。1.neme,type;2.il;3.type;4.bit-wide
    for(int i=0;i<layer_name.size();++i)
    {
        LayerInfo *tempLI = new LayerInfo(layer_name[i],types[i]);
        tempLI->setIl(in[i],out[i],param[i]);
        tempLI->layerID = layer_id[i];
        //tempLI->setBitWide(16);
        layers1.push_back(tempLI);
    }
    //第七步更新界面
    this->refreshList1();
    this->refreshList2();
}


void MainWindow::refreshList1(){


    strList1.clear();
    for(int i=0;i<layers1.size();i++){
        QString temp;
        temp.append(QString::number(i));
        temp.append(" ");
        temp.append(QString::fromStdString(layers1[i]->layerName));
        strList1.append(temp);
    }
    model1 = new QStringListModel(strList1);
    ui->layerList1->setModel(model1);
}

void MainWindow::refreshList2(){
    strList2.clear();
    for(int i=0;i<layers2.size();i++){
        QString temp;
        temp.append(QString::number(i));
        temp.append(" ");
        temp.append(QString::fromStdString(layers2[i]->layerName));
        strList2.append(temp);
    }
    model2 = new QStringListModel(strList2);
    ui->layerList2->setModel(model2);

}

void MainWindow::on_layerList1_doubleClicked(const QModelIndex &index)
{

    int current;
    current = index.row();
    showList1(current);
    int i=0;
    for(;i<this->layers2.size();i++){
        if(layers2[i]->layerID==layers1[current]->layerID)
        {
            //设置选中在2中找到的层
            QModelIndex index2 = model2->index(i);
            ui->layerList2->setCurrentIndex(index2);
            //ui->layerList2->selectionModel()->setCurrentIndex(index2,QItemSelectionModel::Select);
            showList2(i);
            break;
        }
    }
    //当未找到时设置为非选中
    if(i==layers2.size()){
        qDebug()<<"i:"<<i<<" layers2.size:"<<layers2.size();
        QModelIndex index2 = model2->index(-1);
        ui->layerList2->setCurrentIndex(index2);
    }
}

void MainWindow::on_layerList2_doubleClicked(const QModelIndex &index)
{
    int current;
    current = index.row();
    showList2(current);
    int i=0;
    for(;i<this->layers1.size();i++){
        if(layers1[i]->layerID==layers2[current]->layerID)
        {
            //TODO 将List1设置为选中第i项
            QModelIndex index1 = model1->index(i);
            ui->layerList1->setCurrentIndex(index1);
            //ui->layerList2->selectionModel()->setCurrentIndex(index2,QItemSelectionModel::Select);
            showList1(i);
            break;
        }
    }
//2中出现的层肯定能在1中找到原层
//    if(i==layers1.size()){
//        qDebug()<<"i:"<<i<<" layers1.size:"<<layers1.size();
//        QModelIndex index1 = model1->index(-1);
//        ui->layerList1->setCurrentIndex(index1);
//    }
}


void MainWindow::showList1(int current)
{
    //qDebug()<<index.row()<<" "<<index.data();
    ui->layerID1->setText(QString::number(layers1[current]->layerID));
    ui->layerName1->setText(QString::fromStdString(layers1[current]->layerName));

    ui->layerDetails1->clear();
    ui->layerDetails1->append("type: "+QString::fromStdString(layers1[current]->getTypeName()));
    ui->layerDetails1->append("inMaxabs: "+QString::number(layers1[current]->inMaxabs));
    ui->layerDetails1->append("outMaxabs: "+QString::number(layers1[current]->outMaxabs));
    ui->layerDetails1->append("paramMaxabs: "+QString::number(layers1[current]->paramMaxabs));
    ui->layerDetails1->append("inIl: "+QString::number(layers1[current]->inIl));
    ui->layerDetails1->append("outIl: "+QString::number(layers1[current]->outIl));
    ui->layerDetails1->append("paramIl: "+QString::number(layers1[current]->paramIl));
}

void MainWindow::showList2(int current)
{
    //qDebug()<<index.row()<<" "<<index.data();
    //ui->layerID1->setText(QString::number(layers1[current]->layerID));
    //ui->layerName1->setText(QString::fromStdString(layers1[current]->layerName));
    //TODO 将id，name打印到Details2中

    ui->layerDetails2->clear();
    ui->layerDetails2->append("type: "+QString::fromStdString(layers2[current]->getTypeName()));
    ui->layerDetails2->append("inMaxabs: "+QString::number(layers2[current]->inMaxabs));
    ui->layerDetails2->append("outMaxabs: "+QString::number(layers2[current]->outMaxabs));
    ui->layerDetails2->append("paramMaxabs: "+QString::number(layers2[current]->paramMaxabs));
    ui->layerDetails2->append("inIl: "+QString::number(layers2[current]->inIl));
    ui->layerDetails2->append("outIl: "+QString::number(layers2[current]->outIl));
    ui->layerDetails2->append("paramIl: "+QString::number(layers2[current]->paramIl));
}

void MainWindow::on_pushButton_5_clicked()//select layer button
{
    const QModelIndex &index = ui->layerList1->currentIndex();
    qDebug()<<index.row();
    if(index.row() <= 0)return;
    //构造一个新的LayerInfo并添加bit数，还要改层名，层类型
    LayerInfo* temp = new LayerInfo(*layers1[index.row()]);
    temp->bitWide = this->ui->BWSelect->currentText().toInt();
    std::string lname = temp->layerName;
    if(temp->layerType == LayerInfo::CONVOLUTION||temp->layerType == LayerInfo::FULLCONNECTION)
        temp->layerName = lname+"_ris"+std::to_string(temp->bitWide);
    temp->toRistrettoType();

    int ind = 0;
    for(;ind<layers2.size();ind++){
        if(layers1[index.row()]->layerID==layers2[ind]->layerID)
            break;
    }
    if(ind==layers2.size()){
        layers2.push_back(temp);
    }else{
        LayerInfo* toDel = layers2[ind];
        delete toDel;
        layers2[ind] = temp;
    }
    refreshList2();
    QModelIndex index2 = model2->index(ind);
    ui->layerList2->setCurrentIndex(index2);
    showList2(ind);

}


void MainWindow::on_Select_prototxt_clicked()
{

}

void MainWindow::on_pushButton_3_clicked()
{
    //第一步 构造网络
    NetParameter netparam = EditConvolution2DynamicFixedPoint();
    netparam.mutable_state()->set_phase(caffe::TEST);
    //第二步 计算mAp
    //构造map_cal
    mAP_cal* map_cal = new mAP_cal(netparam, weights);
    map_cal->mAP_calc();
    //第三步 输出模型
}
