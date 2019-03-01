
#include "glwidget.h"

#include <QPainter>
#include <QTimer>
#include <QPaintEvent>
#include <QRgb>
#include <QTime>

GLWidget::GLWidget(QWidget *parent)
        : QOpenGLWidget(parent)
{
    elapsed = 0;
    setFixedSize(1200, 800);
    setAutoFillBackground(false);

    m_carImg = QPixmap(":/tracks/car.png");
    m_trackImg = QPixmap(":/tracks/track2.png");
    QImage trackI = m_trackImg.toImage();

    m_map = Eigen::MatrixXi(trackI.height(), trackI.width());
    m_map.setZero();
    for( size_t m = 0; m < m_map.rows(); m++)
    {
        for( size_t n = 0; n < m_map.cols(); n++ )
        {
            QRgb color = trackI.pixel(n,m);
            if( qRed(color) > 60 && qGreen(color) > 60 && qBlue(color) > 60 )
                m_map(m,n) = 1;
        }
    }

    std::shared_ptr<CarFactory> f(new CarFactory(m_map));
    m_evo = new Evolution(1000,200,f);

    m_nextGeneration.start();
}

void GLWidget::animate()
{
    elapsed = (elapsed + qobject_cast<QTimer*>(sender())->interval()) % 1000;
    update();
}

void GLWidget::paintEvent(QPaintEvent *event)
{
    if( m_nextGeneration.elapsed() > 120000 || m_evo->isEpochOver())
    {
        m_evo->breed();
        m_nextGeneration.restart();
    }

    m_evo->doStep();

    QPainter painter;
    painter.begin(this);
    painter.setRenderHint(QPainter::Antialiasing);

    painter.fillRect(event->rect(), QBrush(QColor(64, 32, 64)));

    painter.drawPixmap(0,0,m_trackImg);

    std::vector<SimulationPtr> simRes = m_evo->getSimulationsOrderedByFitness();


    for( size_t k = 0; k < simRes.size(); k++ )
    {
        std::shared_ptr<Car> thisCar = std::dynamic_pointer_cast<Car>( simRes[k] );
        drawCar(&painter, thisCar, Qt::green);
    }

    if( simRes.size() >= 2 ) // overdraw the two best
    {
        std::shared_ptr<Car> bestCar = std::dynamic_pointer_cast<Car>( simRes[0] );
        drawCar(&painter, bestCar, Qt::red);

        std::shared_ptr<Car> secondBestCar = std::dynamic_pointer_cast<Car>( simRes[1] );
        drawCar(&painter, secondBestCar, Qt::yellow);
    }

    auto stat = m_evo->getNumberAliveAndDead();
    QString statText = "Alive: " + QString::number(stat.first) + "   Dead: " + QString::number(stat.second);
    painter.drawText(QPoint(1000,500), statText);

    QString{"Average age: %1 s"}.arg(m_evo->getSimulationsAverageAge(), 0, 'f', 2 );
    painter.drawText(QPoint(1000,530), QString{"Average age: %1"}.arg(m_evo->getSimulationsAverageAge(), 0, 'f', 2 ));

    painter.end();
}

void GLWidget::drawCar(QPainter *painter, std::shared_ptr<Car> car, QColor color)
{
    int carLength = 26;
    int carWidth = 14;

    QMatrix rm;
    rm = rm.rotate(car->getRotationRelativeToInitial());
    QPixmap rotCar = m_carImg.transformed(rm, Qt::SmoothTransformation);

    QPointF carPos( car->getPosition()(0,0),car->getPosition()(1,0) );
    //painter->drawPixmap(carPos - QPointF(carLength/2.0, carWidth/2.0),rotCar);
    painter->setBrush(QBrush(color));

    if( car->isAlive() )
    {
        painter->drawEllipse(carPos, 8, 8);

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
        painter->drawEllipse(carPos, 4, 4);
    }
}



