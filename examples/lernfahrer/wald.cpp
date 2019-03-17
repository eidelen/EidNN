//
// Created by Adrian Schneider on 2019-03-17.
//

#include "wald.h"

Wald::Wald(const QString &name, const QString &rscPath) : Track(name, rscPath), m_anim(0.0)
{
    m_originalMap = createMap(getTrackImg());
}

Wald::~Wald()
{

}

void Wald::draw(QPainter *painter, const std::vector<SimulationPtr > &simRes)
{
    drawMap(painter);

    // draw moving obstacle
    size_t obstWidth = 30;
    size_t obstHeight = 40;
    size_t obstStartPosX = 90;
    size_t obstStartPosY = 330;
    size_t obstaclePos = std::sin(m_anim) * 20.0 + obstStartPosX;

    m_anim = m_anim + 0.04;

    Eigen::MatrixXi newDynMap = m_trackMap->createAllValidMap();

    for( size_t m = 0; m < obstHeight; m++ )
        for( size_t n = 0; n < obstWidth; n++ )
            newDynMap(obstStartPosY+m, obstaclePos+n) = 0;

    m_trackMap->setDynamicMap(newDynMap);

    painter->setBrush(QBrush(Qt::blue));
    painter->drawRect(QRect(obstaclePos,obstStartPosY,obstWidth,obstHeight));

    drawAllCars(painter, simRes);
}

