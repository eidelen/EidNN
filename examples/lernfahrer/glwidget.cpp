
#include "glwidget.h"

#include <QPainter>
#include <QTimer>
#include <QPaintEvent>

GLWidget::GLWidget(QWidget *parent)
        : QOpenGLWidget(parent)
{
    elapsed = 0;
    setFixedSize(1200, 800);
    setAutoFillBackground(false);

    m_carImg = QPixmap(":/tracks/car.png");
    m_trackImg = QPixmap(":/tracks/track1.png");

    m_car.reset( new Car() );
    m_car->setPosition(Eigen::Vector2d(500,150) );
    m_car->setDirection(Eigen::Vector2d(1,0));
    m_car->setRotationSpeed(0.0);
    m_car->setAcceleration(20);
    m_car->setSpeed(10.0);
}

void GLWidget::animate()
{
    elapsed = (elapsed + qobject_cast<QTimer*>(sender())->interval()) % 1000;
    update();
}

void GLWidget::paintEvent(QPaintEvent *event)
{
    m_car->doStep();

    QPainter painter;
    painter.begin(this);
    painter.setRenderHint(QPainter::Antialiasing);

    painter.fillRect(event->rect(), QBrush(QColor(64, 32, 64)));

    painter.drawPixmap(0,0,m_trackImg);

    drawCar(&painter, QPointF(m_car->getPosition()(0,0),m_car->getPosition()(1,0)),  QPointF(m_car->getDirection()(0,0),m_car->getDirection()(1,0)));

    painter.end();
}

void GLWidget::drawCar(QPainter *painter, QPointF pos, QPointF dir)
{
    int carLength = 26;
    int carWidth = 14;

    QMatrix rm;
    rm = rm.rotate(m_car->getRotationRelativeToInitial());
    QPixmap rotCar = m_carImg.transformed(rm, Qt::SmoothTransformation);

    painter->drawPixmap(pos - QPointF(carLength/2.0, carWidth/2.0),rotCar);
}



