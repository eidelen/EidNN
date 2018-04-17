#ifndef WIDGET_H
#define WIDGET_H

#include "network.h"
#include "network_cb.h"
#include "faceDataInput.h"
#include <QMainWindow>
#include <QTimer>
#include <QMutex>
#include <QLineSeries>
#include <QValueAxis>
#include <vector>
#include <memory>
#include <atomic>
#include "opencv2/opencv.hpp"


namespace Ui
{
    class Widget;
}

class Widget : public QMainWindow, public NetworkOperationCallback
{
    Q_OBJECT

public:
    explicit Widget(QWidget* parent = 0);
    ~Widget();

    // NetworkOperationCallback interface
public:
    void networkOperationProgress( const NetworkOperationId &opId, const NetworkOperationStatus &opStatus,
                                   const double &progress );
    void networkTestResults(const double& successRateEuclidean, const double& successRateMaxIdx ,
                            const std::vector<size_t>& failedSamplesIdx);

private:
    void displayTestMNISTImage(const size_t &idx);
    void learn();
    void sameImage();
    void prepareSamplesAndNetwork();
    Eigen::MatrixXd lableToOutputVector( const uint8_t& lable );
    Network::ECostFunction getCurrentSelectedCostFunction();
    void getMinMaxYValue(const QtCharts::QLineSeries* series, const uint &nbrEntries, double& min, double& max);
    void drawImg(const Eigen::MatrixXd& img);

public slots:
    void doNNTesting();
    void doNNLearning();
    void updateUi();
    void loadNN();
    void saveNN();
    void doLive();

signals:
    void readyForTesting();
    void readyForLearning();

private:
    Ui::Widget* ui;
    QtCharts::QLineSeries* m_plotData_classification;
    QtCharts::QLineSeries* m_plotData_L2;
    QtCharts::QValueAxis* m_XAxis;
    QtCharts::QValueAxis* m_YAxis;
    QTimer* m_uiUpdaterTimer;
    QTimer* m_processLiveTimer;
    std::vector<Eigen::MatrixXd> m_batchin;
    std::vector<Eigen::MatrixXd> m_batchout;
    std::vector<Eigen::MatrixXd> m_testin;
    std::vector<Eigen::MatrixXd> m_testout;

    std::vector<Eigen::MatrixXd> m_testDataVisible;

    size_t m_currentIdx;
    std::shared_ptr<Network> m_net;
    std::shared_ptr<Network> m_net_testing;

    // thread safe ui values
    std::atomic<double> m_sr_L2, m_sr_MAX;
    std::atomic<double> m_progress_testing;
    std::atomic<double> m_progress_learning;
    QMutex m_uiLock;
    std::vector<std::size_t> m_failedSamples;

    FaceDataInput* m_data;

    double m_networkError;

    cv::VideoCapture* m_cam;
    cv::CascadeClassifier m_face_cascade;
    cv::CascadeClassifier m_eyes_cascade;
};

#endif // WIDGET_H
