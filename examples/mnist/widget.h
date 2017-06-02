#ifndef WIDGET_H
#define WIDGET_H

#include <QMainWindow>

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
    void showAImage();

private:
    Ui::Widget* ui;
};

#endif // WIDGET_H
