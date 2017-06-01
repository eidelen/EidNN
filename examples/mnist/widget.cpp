#include "widget.h"
#include "ui_widget.h"

#include "mnist/mnist_reader.hpp"

#include <QFile>
#include <vector>

Widget::Widget(QWidget* parent) : QMainWindow(parent), ui(new Ui::Widget)
{
    ui->setupUi(this);
    mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
}

Widget::~Widget()
{
    // Note: Since smartpointers are used, objects get deleted automatically.
    delete ui;
}
