#ifndef CVJOBS_H
#define CVJOBS_H

#include <QObject>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include "opencv2/objdetect.hpp"

using namespace cv;
using namespace cv::dnn;

class CVJobs : public QObject
{
    Q_OBJECT
public:
    explicit CVJobs(QObject *parent = nullptr );


    void init(int num);
    void StopRun();
    void setConfidence(float confidence);

    static void run(Net net = Net(), CascadeClassifier faceCascade = CascadeClassifier());
    static void detectFaceOpenCVDNN(Net net, Mat &frameOpenCVDNN, int num);
    static void detectFaceOpenCVHaar(CascadeClassifier faceCascade, Mat &frameOpenCVHaar, int inHeight=300, int inWidth=0);
    static QImage matToImg(Mat *in_mat);
    static QPixmap MatToPixmap(cv::Mat src);

    static int imgHeight;
    static int objHeight;
private:
    const QString caffeConfigFile = "models/deploy.prototxt";
    const QString caffeWeightFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel";

    const QString tensorflowConfigFile = "models/opencv_face_detector.pbtxt";
    const QString tensorflowWeightFile = "models/opencv_face_detector_uint8.pb";

    const QString faceCascadePath = "models/haarcascade_frontalface_default.xml";


    static float m_confidence;
    constexpr static const double m_scalefactor = 1.0;
    static const int m_Width = 300;
    static const int m_Height = 300;
    static cv::Scalar meanVal;
    static int num;
    static bool m_stopRun;




signals:


public slots:
    void _onMethodChange(int i);
    void _onConfidanceChanged(double d);

};

#endif // CVJOBS_H
