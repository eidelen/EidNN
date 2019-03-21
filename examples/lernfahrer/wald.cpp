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
    Eigen::MatrixXi newDynMap = m_trackMap->createAllValidMap();

    size_t obstStartPosX = 90;
    size_t obstStartPosY = 330;
    size_t movingDist = 20;

    size_t obstaclePos = std::sin(m_anim) * movingDist/2.0 + obstStartPosX;
    m_anim = m_anim + 0.03;
    addDynamicSquare(painter, newDynMap, obstaclePos, obstStartPosY, 40, 40);
    addDynamicSquare(painter, newDynMap, obstaclePos+10, obstStartPosY+40, 20, 20);

    m_trackMap->setDynamicMap(newDynMap);


    drawAllCars(painter, simRes);
}

