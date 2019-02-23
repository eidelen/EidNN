
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
}

void GLWidget::animate()
{
    elapsed = (elapsed + qobject_cast<QTimer*>(sender())->interval()) % 1000;
    update();
}

void GLWidget::paintEvent(QPaintEvent *event)
{
    QPainter painter;
    painter.begin(this);
    painter.setRenderHint(QPainter::Antialiasing);

    painter.fillRect(event->rect(), QBrush(QColor(64, 32, 64)));

    QPixmap pixmap(":/tracks/track1.png");
    painter.drawPixmap(0,0,pixmap);

    drawCar(&painter, QPointF(600,150), QPointF(1,0));

    painter.end();
}

void GLWidget::drawCar(QPainter *painter, QPointF pos, QPointF dir)
{
    int carLength = 26;
    int carWidth = 14;

    QPixmap pixmap(":/tracks/car.png");
    painter->drawPixmap(pos - QPointF(carLength/2.0, carWidth/2.0),pixmap);
}



