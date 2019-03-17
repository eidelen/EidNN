//
// Created by Adrian Schneider on 2019-03-10.
//

#include "track.h"
#include "trackmap.h"
#include "carfactory.h"

Track::Track(const QString &name, const QString &rscPath): m_name(name)
{
    m_trackImg = new QPixmap(rscPath);

    m_trackMap.reset(new TrackMap(createMap(m_trackImg)));
    m_factory.reset(new CarFactory(m_trackMap));
}

Track::~Track()
{
    delete m_trackImg;
}

std::shared_ptr<CarFactory> Track::getFactory() const
{
    return m_factory;
}

QString Track::getName() const
{
    return m_name;
}

QPixmap* Track::getTrackImg() const
{
    return m_trackImg;
}

Eigen::MatrixXi Track::createMap(QPixmap* imgP) const
{
    QImage img = imgP->toImage();
    Eigen::MatrixXi map = Eigen::MatrixXi(img.height(), img.width());
    map.setZero();
    for( size_t m = 0; m < map.rows(); m++)
    {
        for( size_t n = 0; n < map.cols(); n++ )
        {
            QRgb color = img.pixel(n,m);
            if( qRed(color) == 165 && qGreen(color) == 172 && qBlue(color) == 182 ) // Color of the racing track
                map(m,n) = 1;
        }
    }
    return map;
}

void Track::draw(QPainter *painter, const std::vector<SimulationPtr>& simRes)
{
    drawMap(painter);
    drawAllCars(painter, simRes);
}

void Track::drawMap(QPainter *painter)
{
    painter->drawPixmap(0,0,*getTrackImg());
}

void Track::drawAllCars(QPainter *painter, const std::vector<SimulationPtr > &simRes)
{
    for( size_t k = 0; k < simRes.size(); k++ )
    {
        std::shared_ptr<Car> thisCar = std::dynamic_pointer_cast<Car>( simRes[k] );
        drawCar(painter, thisCar, Qt::green);
    }

    // specially mark the two best
    if( simRes.size() >= 0 )
        drawCar(painter, std::dynamic_pointer_cast<Car>( simRes[1] ), Qt::yellow);

    if( simRes.size() >= 1 )
        drawCar(painter, std::dynamic_pointer_cast<Car>( simRes[0] ), Qt::red);
}

void Track::drawCar(QPainter *painter, std::shared_ptr<Car> car, QColor color)
{
    QPointF carPos( car->getPosition()(0,0),car->getPosition()(1,0) );
    painter->setBrush(QBrush(color));

    if( car->isAlive() )
    {
        int carSize = 8;
        painter->drawEllipse(carPos, carSize, carSize);

        // draw distances
        Eigen::MatrixXd distances = car->getMeasuredDistances();
        for (size_t i = 0; i < distances.rows(); i++)
        {
            QPointF distEnd(distances(i, 1), distances(i, 2));
            painter->drawLine(carPos, distEnd);
        }
    }
    else
    {
        int carSizeDead = 3;
        painter->drawEllipse(carPos, carSizeDead, carSizeDead);
    }
}

