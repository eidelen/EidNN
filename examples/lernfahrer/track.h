//
// Created by Adrian Schneider on 2019-03-10.
//

#ifndef EIDNN_TRACK_H
#define EIDNN_TRACK_H

#include "car.h"

#include <QString>
#include <QPixmap>
#include <QPainter>
#include <Eigen/Dense>

class CarFactory;
class TrackMap;

class Track
{
public:
    Track(const QString& name, const QString& rscPath);
    virtual ~Track();

    std::shared_ptr<CarFactory> getFactory() const;
    QString getName() const;
    QPixmap* getTrackImg() const;

    virtual void draw(QPainter *painter, const std::vector<SimulationPtr>& simRes);

protected:
    Eigen::MatrixXi createMap(QPixmap* imgP) const;
    void drawMap(QPainter* painter);
    void drawCar(QPainter* painter, std::shared_ptr<Car> car, QColor color);
    void drawAllCars(QPainter *painter, const std::vector<SimulationPtr>& simRes);
    void addDynamicSquare(QPainter *painter, Eigen::MatrixXi &dynMap, size_t px, size_t py, size_t width, size_t height);

protected:
    QString m_name;
    std::shared_ptr<CarFactory> m_factory;
    std::shared_ptr<TrackMap> m_trackMap;
    QPixmap* m_trackImg;
};


#endif //EIDNN_TRACK_H
