#include "cvjobs.h"
#include <QtConcurrent/QtConcurrent>
#include <QImage>
#include <QDebug>
#include <QPixmap>
#include <unistd.h>
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
int CVJobs::num = 0;
cv::Scalar CVJobs::meanVal = cv::Scalar(104.0, 177.0, 123.0);
float CVJobs::m_confidence = 0.7;
bool CVJobs::m_stopRun = false;
int CVJobs::imgHeight = 0;
int CVJobs::objHeight = 0;
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
CVJobs::CVJobs(QObject *parent) : QObject(parent)
{


}
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
void CVJobs::init(int num)
{

    Net net;
    CascadeClassifier faceCascade;

    this->num = num;
    // 0 -> dnn tf , 1 -> dnn caffe , 2 -> harr
    switch (num) {
    case 0:
    {
        net = cv::dnn::readNetFromTensorflow(tensorflowWeightFile.toStdString(), tensorflowConfigFile.toStdString());
        break;
    }
    case 1:
    {
        net = cv::dnn::readNetFromCaffe(caffeConfigFile.toStdString(), caffeWeightFile.toStdString());
        break;
    }
    case 2:
    {
        if(!faceCascade.load(faceCascadePath.toStdString()))
        {
            printf("--(!)Error loading face cascade\n");
            return;
        }

        break;
    }
    default:
        break;
    }

    net.setPreferableBackend(DNN_TARGET_CPU);

    StopRun();


    //new thread
    QtConcurrent::run(CVJobs::run,net,faceCascade);



}
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
void CVJobs::run( Net net , CascadeClassifier faceCascade)
{
    m_stopRun = true;

    static VideoCapture source;
    if(!source.isOpened())
        source.open(0, CAP_V4L);
    cv::Mat frame;



    double tt_opencvDNN = 0;
    double fpsOpencv = 0;

    while (m_stopRun)
    {
        source.grab();
        source >> frame;
        imgHeight = frame.rows;
        if (frame.empty())
            break;

        double t = cv::getTickCount();
        if(num == 2)
            detectFaceOpenCVHaar(faceCascade, frame,m_Height,m_Width);
        else
            detectFaceOpenCVDNN(net,frame, num);

        tt_opencvDNN = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
        fpsOpencv = 1/tt_opencvDNN;


//        if(num == 2)
//        {
//            putText(frame, format("OpenCV HAAR ; FPS = %.2f",fpsOpencv), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.3, Scalar(0, 0, 255), 4);
//            imshow("OpenCV - HAAR Face Detection", frame);
//        }
//        else
//        {
//            putText(frame, format("OpenCV DNN %s FPS = %.2f", QString("CPU").toStdString().c_str(), fpsOpencv), Point(5, 50), FONT_HERSHEY_SIMPLEX, 1.3, Scalar(0, 0, 255), 4);
//            imshow("OpenCV - DNN Face Detection",frame);
//        }

//        int k = waitKey(5);
//        if(k == 27)
//        {
//            destroyAllWindows();
//            break;
//        }

//        if(!m_stopRun)
//        {
//            destroyAllWindows();
//            return;
//        }
        usleep(1000000);
    }
    qDebug()<<"exited";
}
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
void CVJobs::detectFaceOpenCVDNN(Net net, Mat &frameOpenCVDNN, int num)
{
    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;

    cv::Mat inputBlob;
    switch (num) {
    case 0:
    {
        inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, m_scalefactor, cv::Size(m_Width, m_Height), meanVal, true, false);
        break;
    }
    case 1:
    {
        inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, m_scalefactor, cv::Size(m_Width, m_Height), meanVal, false, false);
        break;
    }
    }

    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for(int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);
        if(confidence > m_confidence)
        {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

            objHeight = y2 - y1;

            cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),2, 4);
        }
    }


}
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
void CVJobs::detectFaceOpenCVHaar(CascadeClassifier faceCascade, Mat &frameOpenCVHaar, int inHeight, int inWidth)
{
    //gray scale
    int frameHeight = frameOpenCVHaar.rows;
    int frameWidth = frameOpenCVHaar.cols;
    if (!inWidth)
        inWidth = (int)((frameWidth / (float)frameHeight) * inHeight);

    float scaleHeight = frameHeight / (float)inHeight;
    float scaleWidth = frameWidth / (float)inWidth;

    Mat frameOpenCVHaarSmall, frameGray;
    resize(frameOpenCVHaar, frameOpenCVHaarSmall, Size(inWidth, inHeight));
    cvtColor(frameOpenCVHaarSmall, frameGray, COLOR_BGR2GRAY);

    std::vector<Rect> faces;
    faceCascade.detectMultiScale(frameGray, faces);

    for ( size_t i = 0; i < faces.size(); i++ )
    {
        int x1 = (int)(faces[i].x * scaleWidth);
        int y1 = (int)(faces[i].y * scaleHeight);
        int x2 = (int)((faces[i].x + faces[i].width) * scaleWidth);
        int y2 = (int)((faces[i].y + faces[i].height) * scaleHeight);

        objHeight = y2 - y1;

        rectangle(frameOpenCVHaar, Point(x1, y1), Point(x2, y2), Scalar(0,255,0), (int)(frameHeight/150.0), 4);
    }
}
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
void CVJobs::setConfidence(float confidence)
{
    m_confidence = confidence;
}
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
void CVJobs::StopRun()
{
    m_stopRun = false;
}
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
void CVJobs::_onMethodChange(int i)
{
    init(i);
}
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
void CVJobs::_onConfidanceChanged(double d)
{
    m_confidence = d;
}
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
QImage CVJobs::matToImg(Mat *in_mat)
{
    Mat temp;

    if(in_mat)
    {
        cvtColor(*in_mat, temp,COLOR_BGR2RGB);
        QImage return_img = QImage((const uchar*)temp.data,temp.cols,temp.rows,temp.step,QImage::Format_RGB888);
        return_img.bits();
        return return_img;
    }
    else
        return QImage();
}
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
QPixmap CVJobs::MatToPixmap(cv::Mat src)
{
    QImage::Format format=QImage::Format_Grayscale8;
    int bpp=src.channels();
    if(bpp==3)format=QImage::Format_RGB888;
    QImage img(src.cols,src.rows,format);
    uchar *sptr,*dptr;
    int linesize=src.cols*bpp;
    for(int y=0;y<src.rows;y++){
        sptr=src.ptr(y);
        dptr=img.scanLine(y);
        memcpy(dptr,sptr,linesize);
    }
    if(bpp==3)return QPixmap::fromImage(img.rgbSwapped());
    return QPixmap::fromImage(img);
}
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------

