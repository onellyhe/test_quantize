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
