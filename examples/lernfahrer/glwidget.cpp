
#include "glwidget.h"
#include "strange.h"
#include "carfactory.h"

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

    m_currentTrack->draw(&painter, simRes);

    // draw text
    QFont font = painter.font() ;
    font.setPointSize(25);
    painter.setFont(font);
    painter.setPen(QColor(255,255,255));
    auto stat = m_evo->getNumberAliveAndDead();
    int menuYOffset = 550;
    int menuXOffset = 940;
    painter.drawText(QPoint(menuXOffset,menuYOffset), QString{"Alive: %1   Dead: %2"}.arg(stat.first).arg(stat.second));
    painter.drawText(QPoint(menuXOffset,menuYOffset+30), QString{"Average age: %1"}.arg(m_evo->getSimulationsAverageAge(), 0, 'f', 2 ));
    painter.drawText(QPoint(menuXOffset,menuYOffset+60), QString{"Epoch: %1"}.arg(m_evo->getNumberOfEpochs()));
    painter.drawText(QPoint(menuXOffset,menuYOffset+90), QString{"FPS: %1"}.arg(m_evo->getSimulationStepsPerSecond(), 0, 'f', 2 ));

    if( simRes.size() > 0 )
    {
        std::shared_ptr<Car> leader = std::dynamic_pointer_cast<Car>( simRes[0] );

        font.setPointSize(20);
        painter.setFont(font);
        painter.drawText(QPoint(menuXOffset,menuYOffset+140), QString{"Leader: %1"}.arg(leader->isAlive() ? "Driving" : "Crashed"));
        painter.drawText(QPoint(menuXOffset,menuYOffset+160), QString{"Distance: %1 px"}.arg((int)leader->getFitness()));
        painter.drawText(QPoint(menuXOffset,menuYOffset+180), QString{"Speed: %1 px/s"}.arg((int)leader->getSpeed()));
        painter.drawText(QPoint(menuXOffset,menuYOffset+200), QString{"Acceleration : %1 px/s^2"}.arg((int)leader->getAcceleration()));
        painter.drawText(QPoint(menuXOffset,menuYOffset+220), QString{"Rotation : %1 °/s"}.arg((int)leader->getRotationSpeed()));

    }

    painter.end();
}

void GLWidget::doNewEpoch()
{
    m_evo->killAllSimulations();
}

void GLWidget::startRace(std::shared_ptr<Track> t)
{
    m_doSimulation = false;

    m_currentTrack = t;

    if( m_evo )
    {
        m_evo->killAllSimulations();
        m_evo->resetFactory(t->getFactory());
    }
    else
    {
        m_evo.reset( new Evolution(1200, 100, t->getFactory(), 12) );
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
    m_tracks.push_back(std::shared_ptr<Track>(new Track("Nabu", ":/tracks/track2.png")));
    m_tracks.push_back(std::shared_ptr<Track>(new Track("Tartaros", ":/tracks/track3.png")));
    m_tracks.push_back(std::shared_ptr<Track>(new Track("Wald", ":/tracks/track4.png")));
    m_tracks.push_back(std::shared_ptr<Track>(new Strange("Strange", ":/tracks/track5.png")));

    m_currentTrackIdx = 0;
    startRace(m_tracks.at(m_currentTrackIdx));
}

void GLWidget::mutationRateChanged(double mutRate)
{
    if( m_evo )
        m_evo->setMutationRate(mutRate);
}

void GLWidget::save()
{
    m_evo->save("a.net", "b.net");
}

void GLWidget::load()
{
    m_evo->load("a.net", "b.net");
}

