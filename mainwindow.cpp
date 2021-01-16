#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QtConcurrent/QtConcurrent>


//cv::Mat *MainWindow::frame;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{

    ui->setupUi(this);

    job = new CVJobs();
    timer = new QTimer;

    connect(ui->combo_input, SIGNAL(currentIndexChanged(int)) , job , SLOT(_onMethodChange(int)),Qt::QueuedConnection);
    connect(ui->dspin_confidance, SIGNAL(valueChanged(double)) , job , SLOT(_onConfidanceChanged(double)),Qt::QueuedConnection);
    job->init(0);


    connect(timer,&QTimer::timeout,this,&MainWindow::_onTimeout);
    timer->start(100);

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::_onTimeout()
{
    if(CVJobs::objHeight == 0 || CVJobs::imgHeight == 0)
    {
        ui->edt_distance->setText("");
        return;
    }
    qDebug()<<"CVJobs::objHeight"<< CVJobs::objHeight << "CVJobs::imgHeight" << CVJobs::imgHeight;

    ui->edt_distance->setText(QString::number((4*ui->spin_height->value() * (CVJobs::imgHeight))/(CVJobs::objHeight * 3.33)));
}


