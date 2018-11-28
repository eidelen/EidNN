#ifndef WIDGET_H
#define WIDGET_H

#include "network.h"
#include "network_cb.h"
#include "mnistDataInput.h"
#include <QMainWindow>
#include <QTimer>
#include <QMutex>
#include <QLineSeries>
#include <QValueAxis>
#include <vector>
#include <memory>
#include <atomic>

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
                                   const double &progress, const int& userId );
    void networkTestResults(const double& successRateEuclidean, const double& successRateMaxIdx ,
                            const double& averageCost,
                            const std::vector<size_t>& failedSamplesIdx, const int& userId);

private:
    bool loadMNISTSample( const std::vector<std::vector<double>>& imgSet, const std::vector<uint8_t>& lableSet,
                          const size_t& idx, Eigen::MatrixXd& img, uint8_t& lable);
    void displayTestMNISTImage(const size_t &idx);
    void learn();
    void sameImage();
    void prepareSamples();
    Eigen::MatrixXd lableToOutputVector( const uint8_t& lable );
    Network::ECostFunction getCurrentSelectedCostFunction();
    void getMinMaxYValue(const QtCharts::QLineSeries* series, const uint &nbrEntries, double& min, double& max);

public slots:
    void doNNTesting();
    void doNNValidation();
    void doNNLearning();
    void updateUi();
    void loadNN();
    void saveNN();
    void setRegularizationFunction();

signals:
    void readyForValidation();
    void readyForTrainingTesting();
    void readyForLearning();

private:
    Ui::Widget* ui;
    QtCharts::QLineSeries* m_plotData_classification;
    QtCharts::QLineSeries* m_trainingSuccess;
    QtCharts::QValueAxis* m_XAxis;
    QtCharts::QValueAxis* m_YAxis;

    QtCharts::QLineSeries* m_testSetCost;
    QtCharts::QValueAxis* m_TCXAxis;
    QtCharts::QValueAxis* m_TCYAxis;

    QtCharts::QLineSeries* m_trainingSetCost;
    QtCharts::QValueAxis* m_RCXAxis;
    QtCharts::QValueAxis* m_RCYAxis;

    QTimer* m_uiUpdaterTimer;
    std::vector<Eigen::MatrixXd> m_batchin;
    std::vector<Eigen::MatrixXd> m_batchout;
    std::vector<Eigen::MatrixXd> m_testin;
    std::vector<Eigen::MatrixXd> m_testout;
    size_t m_currentIdx;
    std::shared_ptr<Network> m_net;
    std::shared_ptr<Network> m_net_validation;
    std::shared_ptr<Network> m_net_training_testing;

    // thread safe ui values
    std::atomic<double> m_sr_L2, m_sr_MAX;
    std::atomic<double> m_progress_training_testing;
    std::atomic<double> m_progress_learning;
    std::atomic<double> m_progress_validation;
    QMutex m_uiLock;
    std::vector<std::size_t> m_failedSamples;

    MnistDataInput* m_data;

    const int NETID_TRAINING{0};
    const int NETID_TRAINING_TESTING{1};
    const int NETID_VALIDATION{2};
};

#endif // WIDGET_H
