
#include "window.h"
#include "glwidget.h"

#include <QLayout>
#include <QLabel>
#include <QTimer>
#include <QPushButton>
#include <QDoubleSpinBox>

Window::Window()
{
    setWindowTitle(tr("EidNN Lernfahrer"));

    GLWidget *openGL = new GLWidget(this);

    QPushButton* nextEpochBtn = new QPushButton("New Epoch");
    QPushButton* nextTrackBtn = new QPushButton("Next");
    QPushButton* saveBtn = new QPushButton("Save");
    QPushButton* loadBtn = new QPushButton("Load");
    QLabel* mutationRateLabel = new QLabel("Mutation Rate:");
    QDoubleSpinBox* mutationRateSpinBox = new QDoubleSpinBox();
    mutationRateSpinBox->setMinimum(0.0); mutationRateSpinBox->setMaximum(1.0);
    mutationRateSpinBox->setSingleStep(0.005); mutationRateSpinBox->setValue(0.05);


    QHBoxLayout* btnLayout = new QHBoxLayout();
    btnLayout->addWidget(nextEpochBtn);
    btnLayout->addWidget(nextTrackBtn);
    btnLayout->addWidget(saveBtn);
    btnLayout->addWidget(loadBtn);
    btnLayout->addWidget(mutationRateLabel);
    btnLayout->addWidget(mutationRateSpinBox);

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
    connect(saveBtn, SIGNAL (released()),openGL, SLOT (save()));
    connect(loadBtn, SIGNAL (released()),openGL, SLOT (load()));
    connect(mutationRateSpinBox, SIGNAL(valueChanged(double)), openGL, SLOT( mutationRateChanged(double)));
}