//
// Created by Adrian Schneider on 2019-03-10.
//

#include "track.h"

Track::Track(const QString &name, const QString &rscPath): m_name(name)
{
    m_trackImg = new QPixmap(rscPath);
    m_factory.reset(new CarFactory(createMap(m_trackImg)));
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

void Track::animate(QPainter *painter)
{

}
