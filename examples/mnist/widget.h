#ifndef WIDGET_H
#define WIDGET_H

#include "network.h"
#include "mnist/mnist_reader.hpp"
#include <QMainWindow>
#include <vector>

namespace Ui
{
    class Widget;
}

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

private:
    Ui::Widget* ui;
    mnist::MNIST_dataset< std::vector, std::vector<double>, uint8_t> m_mnist;
    size_t m_currentIdx;
};

#endif // WIDGET_H
