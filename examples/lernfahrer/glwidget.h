
#ifndef EIDNN_GLWIDGET_H
#define EIDNN_GLWIDGET_H

#include "car.h"
#include "evolution.h"
#include "track.h"

#include <QOpenGLWidget>
#include <QTime>
#include <vector>

class Helper;

class GLWidget : public QOpenGLWidget
{
Q_OBJECT

public:
    GLWidget(QWidget *parent);

public slots:
    void animate();
    void doNewEpoch();
    void nextTrack();
    void mutationRateChanged(double mutRate);

private:
    void startRace(std::shared_ptr<Track> t);
    void initTracks();

protected:
    void paintEvent(QPaintEvent *event) override;
    void drawCar(QPainter* painter, std::shared_ptr<Car> car, QColor color);

private:
    int elapsed;
    std::shared_ptr<Car> m_car;


    std::shared_ptr<Evolution> m_evo;
    Eigen::MatrixXi m_map;
    std::atomic_bool m_doSimulation;
    std::vector<std::shared_ptr<Track>> m_tracks;
    std::shared_ptr<Track> m_currentTrack;
    size_t m_currentTrackIdx;
};

#endif //EIDNN_GLWIDGET_H
