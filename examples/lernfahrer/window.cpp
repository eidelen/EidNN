
#include "window.h"
#include "glwidget.h"

#include <QLayout>
#include <QLabel>
#include <QTimer>
#include <QPushButton>

Window::Window()
{
    setWindowTitle(tr("EidNN Lernfahrer"));

    GLWidget *openGL = new GLWidget(this);

    QPushButton* nextEpochBtn = new QPushButton("New Epoch");

    QVBoxLayout* layout = new QVBoxLayout;
    layout->addWidget(nextEpochBtn);
    layout->addWidget(openGL);

    setLayout(layout);


    // connections

    QTimer *timer = new QTimer(this);
    connect(timer, &QTimer::timeout, openGL, &GLWidget::animate);
    timer->start(40);

    connect(nextEpochBtn, SIGNAL (released()),openGL, SLOT (doNewEpoch()));

}