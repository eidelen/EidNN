
#include "glwidget.h"

#include <QPainter>
#include <QTimer>
#include <QPaintEvent>
#include <QRgb>

GLWidget::GLWidget(QWidget *parent)
        : QOpenGLWidget(parent)
{
    elapsed = 0;
    setFixedSize(1200, 800);
    setAutoFillBackground(false);

    m_carImg = QPixmap(":/tracks/car.png");
    m_trackImg = QPixmap(":/tracks/track1.png");
    QImage trackI = m_trackImg.toImage();

    m_map = Eigen::MatrixXi(trackI.height(), trackI.width());
    m_map.setOnes();
    for( size_t m = 0; m < m_map.rows(); m++)
    {
        for( size_t n = 0; n < m_map.cols(); n++ )
        {
            QRgb color = trackI.pixel(n,m);
            if( qRed(color) < 10 && qGreen(color) < 10 && qBlue(color) < 10)
                m_map(m,n) = 0;
        }
    }

    std::shared_ptr<CarFactory> f(new CarFactory(m_map));
    m_evo = new Evolution(10,200,f);
}

void GLWidget::animate()
{
    elapsed = (elapsed + qobject_cast<QTimer*>(sender())->interval()) % 1000;
    update();
}

void GLWidget::paintEvent(QPaintEvent *event)
{
    m_evo->doStep();

    QPainter painter;
    painter.begin(this);
    painter.setRenderHint(QPainter::Antialiasing);

    painter.fillRect(event->rect(), QBrush(QColor(64, 32, 64)));

    painter.drawPixmap(0,0,m_trackImg);

    std::vector<SimulationPtr> simRes = m_evo->getSimulationsOrderedByFitness();
    for( SimulationPtr c : simRes )
    {
        std::shared_ptr<Car> thisCar = std::dynamic_pointer_cast<Car>( c );
        drawCar(&painter, thisCar);
    }

    painter.end();
}

void GLWidget::drawCar(QPainter *painter, std::shared_ptr<Car> car)
{
    int carLength = 26;
    int carWidth = 14;

    QMatrix rm;
    rm = rm.rotate(car->getRotationRelativeToInitial());
    QPixmap rotCar = m_carImg.transformed(rm, Qt::SmoothTransformation);


    QPointF carPos( car->getPosition()(0,0),car->getPosition()(1,0) );
    painter->drawPixmap(carPos - QPointF(carLength/2.0, carWidth/2.0),rotCar);


    // draw distances
    Eigen::MatrixXd distances = car->getMeasuredDistances();
    for( size_t i = 0; i < distances.rows(); i++ )
    {
        QPointF distEnd( distances(i,1), distances(i,2) );
        painter->drawLine( carPos, distEnd );
    }
}



