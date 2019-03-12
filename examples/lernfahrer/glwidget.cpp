
#include "glwidget.h"
#include "strange.h"

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
    painter.setPen(QColor(39,75,122));
    auto stat = m_evo->getNumberAliveAndDead();
    painter.drawText(QPoint(900,650), QString{"Alive: %1   Dead: %2"}.arg(stat.first).arg(stat.second));
    painter.drawText(QPoint(900,680), QString{"Average age: %1"}.arg(m_evo->getSimulationsAverageAge(), 0, 'f', 2 ));
    painter.drawText(QPoint(900,710), QString{"Epoch: %1"}.arg(m_evo->getNumberOfEpochs()));
    painter.drawText(QPoint(900,740), QString{"FPS: %1"}.arg(m_evo->getSimulationStepsPerSecond(), 0, 'f', 2 ));

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

