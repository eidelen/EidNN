//
// Created by Adrian Schneider on 2019-03-12.
//

#include "strange.h"

Strange::Strange(const QString &name, const QString &rscPath) : Track(name, rscPath), m_anim(0.0)
{
    m_originalMap = createMap(getTrackImg());
}

Strange::~Strange()
{

}

void Strange::draw(QPainter *painter, const std::vector<SimulationPtr > &simRes)
{
    drawMap(painter);

    // draw moving obstacle
    size_t obstStartPosX = 650;
    size_t obstStartPosY = 310;
    size_t obstaclePos = std::sin(m_anim) * 120.0 + obstStartPosY;

    m_anim = m_anim + 0.02;

    Eigen::MatrixXi newDynMap = m_trackMap->createAllValidMap();

    for( size_t m = 0; m < 100; m++ )
        for( size_t n = 0; n < 30; n++ )
            newDynMap(obstaclePos+m, obstStartPosX+n) = 0;

    m_trackMap->setDynamicMap(newDynMap);

    painter->setBrush(QBrush(Qt::blue));
    painter->drawRect(QRect(obstStartPosX,obstaclePos,30,100));

    drawAllCars(painter, simRes);
}
