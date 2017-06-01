#include "widget.h"
#include "ui_widget.h"

#include <QFile>

Widget::Widget(QWidget* parent) : QMainWindow(parent), ui(new Ui::Widget)
{
    ui->setupUi(this);
}

Widget::~Widget()
{
    // Note: Since smartpointers are used, objects get deleted automatically.
    delete ui;
}
