//
// Created by Adrian Schneider on 2019-03-13.
//

#include "trackmap.h"


TrackMap::TrackMap(const Eigen::MatrixXi& map): m_dynamicMapSet(false)
{
    m_map = computeDistanceMap( map );

}

TrackMap::~TrackMap()
{

}

void TrackMap::resetMap(const Eigen::MatrixXi &map)
{
    m_map = map;
}

const Eigen::MatrixXi &TrackMap::getMap() const
{
    return m_map;
}

bool TrackMap::isPositionValid(const Eigen::Vector2d &pos) const
{
    bool valid = isPositionValid(m_map, pos);

    if( valid && m_dynamicMapSet )
        valid = isPositionValid(m_dynamicMap, pos);

    return valid;
}

bool TrackMap::isPositionValid(const Eigen::MatrixXi& map, const Eigen::Vector2d &pos) const
{
    if(pos(0) < 0.0 ||  pos(0) > map.cols()-1 || pos(1) < 0.0 ||  pos(1) > map.rows()-1)
        return false;

    return map( std::ceil(pos(1)) , std::ceil(pos(0))) > 0;
}

void TrackMap::setDynamicMap(const Eigen::MatrixXi &map)
{
    m_dynamicMapSet = true;
    m_dynamicMap = map;
}

void TrackMap::clearDynamicMap()
{
    m_dynamicMapSet = false;
}

Eigen::MatrixXi TrackMap::createAllValidMap() const
{
    return Eigen::MatrixXi::Ones(m_map.rows(), m_map.cols());
}

Eigen::MatrixXi TrackMap::computeDistanceMap( const Eigen::MatrixXi& map ) const
{
    Eigen::MatrixXi dMap = map;

    for( size_t m = 0; m < dMap.rows(); m++ )
    {
        for( size_t n = 0; n < dMap.cols(); n++ )
        {
            Eigen::Vector2d p( n, m );
            if( isPositionValid(map, p) )
            {
                //todo: lets find an approximate smallest distance to the border
                int r = 1;
                bool allValid = true;
                for( ; r < std::min(map.rows(), map.cols() ) && allValid; )
                {
                    // upper h line
                    for( int i = -r; i <= r && allValid; i++ )
                        allValid = isPositionValid(Eigen::Vector2d(n + i, m - r));

                    if(allValid)
                        r++;

                }

                dMap(m, n) = r;

            }
            else
            {
                dMap(m, n) = 0;
            }
        }
    }

    return dMap;
}
