
#ifndef EIDNN_GLWIDGET_H
#define EIDNN_GLWIDGET_H

#include <QOpenGLWidget>

class Helper;

class GLWidget : public QOpenGLWidget
{
Q_OBJECT

public:
    GLWidget(QWidget *parent);

public slots:
    void animate();

protected:
    void paintEvent(QPaintEvent *event) override;
    void drawCar(QPainter* painter, QPointF pos, QPointF dir);

private:
    int elapsed;
};

#endif //EIDNN_GLWIDGET_H
