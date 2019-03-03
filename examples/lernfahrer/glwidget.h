
#ifndef EIDNN_GLWIDGET_H
#define EIDNN_GLWIDGET_H

#include "car.h"
#include "evolution.h"

#include <QOpenGLWidget>
#include <QPixmap>
#include <QTime>

class Helper;

class GLWidget : public QOpenGLWidget
{
Q_OBJECT

public:
    GLWidget(QWidget *parent);

    enum Track
    {
        Track1,
        Track2,
        Track3
    };

public slots:
    void animate();
    void doNewEpoch();
    void nextTrack();

private:
    Eigen::MatrixXi createMap(const QPixmap &img) const;
    void initTrack(Track t);

protected:
    void paintEvent(QPaintEvent *event) override;
    void drawCar(QPainter* painter, std::shared_ptr<Car> car, QColor color);

private:
    int elapsed;
    std::shared_ptr<Car> m_car;
    QPixmap m_trackImg;
    QPixmap m_carImg;

    Evolution* m_evo;
    Eigen::MatrixXi m_map;

    std::atomic_bool m_doSimulation;
};

#endif //EIDNN_GLWIDGET_H
