
#ifndef EIDNN_GLWIDGET_H
#define EIDNN_GLWIDGET_H

#include "car.h"
#include "evolution.h"

#include <QOpenGLWidget>
#include <QPixmap>

class Helper;

class GLWidget : public QOpenGLWidget
{
Q_OBJECT

public:
    GLWidget(QWidget *parent);

public slots:
    void animate();

protected:
    void paintEvent(QPaintEvent *event) override;
    void drawCar(QPainter* painter, std::shared_ptr<Car> car);

private:
    int elapsed;
    std::shared_ptr<Car> m_car;
    QPixmap m_trackImg;
    QPixmap m_carImg;

    Evolution* m_evo;
    Eigen::MatrixXi m_map;

};

#endif //EIDNN_GLWIDGET_H
