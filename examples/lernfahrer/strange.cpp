//
// Created by Adrian Schneider on 2019-03-12.
//

#include "strange.h"

Strange::Strange(const QString &name, const QString &rscPath) : Track(name, rscPath)
{
    m_originalMap = createMap(getTrackImg());
}

Strange::~Strange()
{

}

void Strange::draw(QPainter *painter, const std::vector<SimulationPtr > &simRes)
{
    painter->drawPixmap(0,0,*getTrackImg());

    /*
    painter->setBrush(QBrush(Qt::blue));
    painter->drawRect(QRect(200,200,50,200));

    m_factory->getMap().setOnes();
     */


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
