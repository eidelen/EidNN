#ifndef WIDGET_H
#define WIDGET_H

#include "network.h"
#include <QMainWindow>
#include <QThread>
#include <vector>

namespace Ui
{
    class Widget;
}


struct NNSample
{
    Eigen::MatrixXd input;
    Eigen::MatrixXd normalizedinput;
    Eigen::MatrixXd output;
    uint8_t lable;
};

class Widget : public QMainWindow
{
    Q_OBJECT

public:
    explicit Widget(QWidget* parent = 0);
    ~Widget();

private:
    bool loadMNISTSample( const std::vector<std::vector<double>>& imgSet, const std::vector<uint8_t>& lableSet,
                          const size_t& idx, Eigen::MatrixXd& img, uint8_t& lable);
    void displayMNISTImage(const size_t &idx);
    void learn();
    void sameImage();
    void prepareSamples();
    Eigen::MatrixXd lableToOutputVector( const uint8_t& lable );

private:
    Ui::Widget* ui;
    std::vector<NNSample> m_trainingSet;
    std::vector<NNSample> m_testingSet;
    size_t m_currentIdx;

};

#endif // WIDGET_H
