
#include "glwidget.h"

#include <QPainter>
#include <QTimer>
#include <QPaintEvent>
#include <QRgb>

GLWidget::GLWidget(QWidget *parent)
        : QOpenGLWidget(parent), m_evo(nullptr)
{
    elapsed = 0;
    setFixedSize(1200, 800);
    setAutoFillBackground(false);

    initTracks();
    m_currentTrackIdx = 0;
    startRace(m_tracks.at(m_currentTrackIdx));
}

void GLWidget::animate()
{
    elapsed = (elapsed + qobject_cast<QTimer*>(sender())->interval()) % 1000;
    update();
}

void GLWidget::paintEvent(QPaintEvent *event)
{
    // update simulation
    if( m_doSimulation )
    {
        if (m_evo->isEpochOver())
            m_evo->breed();
        
        m_evo->doStep();
    }

    std::vector<SimulationPtr> simRes = m_evo->getSimulationsOrderedByFitness();

    // draw the simulation

    QPainter painter;
    painter.begin(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.fillRect(event->rect(), QBrush(QColor(64, 32, 64)));
    painter.drawPixmap(0,0,m_trackImg);

    for( size_t k = 0; k < simRes.size(); k++ )
    {
        std::shared_ptr<Car> thisCar = std::dynamic_pointer_cast<Car>( simRes[k] );
        drawCar(&painter, thisCar, Qt::green);
    }

    // specially mark the two best
    if( simRes.size() >= 0 )
        drawCar(&painter, std::dynamic_pointer_cast<Car>( simRes[1] ), Qt::yellow);

    if( simRes.size() >= 1 )
        drawCar(&painter, std::dynamic_pointer_cast<Car>( simRes[0] ), Qt::red);


    // draw text
    QFont font = painter.font() ;
    font.setPointSize(25);
    painter.setFont(font);
    painter.setPen(QColor(39,75,122));
    auto stat = m_evo->getNumberAliveAndDead();
    painter.drawText(QPoint(900,650), QString{"Alive: %1   Dead: %2"}.arg(stat.first).arg(stat.second));
    painter.drawText(QPoint(900,680), QString{"Average age: %1"}.arg(m_evo->getSimulationsAverageAge(), 0, 'f', 2 ));
    painter.drawText(QPoint(900,710), QString{"Epoch: %1"}.arg(m_evo->getNumberOfEpochs()));
    painter.drawText(QPoint(900,740), QString{"FPS: %1"}.arg(m_evo->getSimulationStepsPerSecond(), 0, 'f', 2 ));

    painter.end();
}

void GLWidget::drawCar(QPainter *painter, std::shared_ptr<Car> car, QColor color)
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

void GLWidget::doNewEpoch()
{
    m_evo->killAllSimulations();
}

Eigen::MatrixXi GLWidget::createMap(const QPixmap &imgP) const
{
    QImage img = imgP.toImage();
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

void GLWidget::startRace(GLWidget::Track t)
{
    m_doSimulation = false;
    m_trackImg = QPixmap(t.rscPath);
    m_map = createMap(m_trackImg);

    std::shared_ptr<CarFactory> f(new CarFactory(m_map));

    if( m_evo )
    {
        m_evo->killAllSimulations();
        m_evo->resetFactory(f);
    }
    else
    {
        m_evo.reset( new Evolution(1200, 200, f, 12) );
    }

    m_doSimulation = true;
}

void GLWidget::nextTrack()
{
    m_currentTrackIdx++;
    if( m_currentTrackIdx >= m_tracks.size() )
        m_currentTrackIdx = 0;

    startRace(m_tracks.at(m_currentTrackIdx));
}

void GLWidget::initTracks()
{
    Track t2 = {.name = QString("Nabu"), .rscPath = QString(":/tracks/track2.png")};
    Track t3 = {.name = QString("Tartaros"), .rscPath = QString(":/tracks/track3.png")};
    Track t4 = {.name = QString("Wald"), .rscPath = QString(":/tracks/track4.png")};
    m_tracks.push_back(t2);
    m_tracks.push_back(t3);
    m_tracks.push_back(t4);
}

void GLWidget::mutationRateChanged(double mutRate)
{
    if( m_evo )
        m_evo->setMutationRate(mutRate);
}

