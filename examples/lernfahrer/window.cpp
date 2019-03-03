
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
    QPushButton* nextTrackBtn = new QPushButton("Next");

    QHBoxLayout* btnLayout = new QHBoxLayout();
    btnLayout->addWidget(nextEpochBtn);
    btnLayout->addWidget(nextTrackBtn);

    QVBoxLayout* layout = new QVBoxLayout;
    layout->addLayout(btnLayout);
    layout->addWidget(openGL);

    setLayout(layout);


    // connections

    QTimer *timer = new QTimer(this);
    connect(timer, &QTimer::timeout, openGL, &GLWidget::animate);
    timer->start(10);

    connect(nextEpochBtn, SIGNAL (released()),openGL, SLOT (doNewEpoch()));
    connect(nextTrackBtn, SIGNAL (released()),openGL, SLOT (nextTrack()));
}