#include "widget.h"
#include "ui_widget.h"
#include "layer.h"
#include "helpers.h"
#include <limits>
#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

#include <QPixmap>
#include <QImage>
#include <QMutexLocker>
#include <QStringListModel>
#include <QChart>
#include <QChartView>
#include <QFileDialog>

Widget::Widget(QWidget* parent) : QMainWindow(parent), ui(new Ui::Widget),
    m_sr_L2(0), m_sr_MAX(0), m_progress_testing(0), m_progress_learning(0)
{
    ui->setupUi(this);


    // crate chart
    QtCharts::QChart *chart = new QtCharts::QChart( );
    chart->legend()->hide();
    m_plotData_classification = new QtCharts::QLineSeries( );
    m_plotData_classification->append(0,0);
    m_plotData_L2 = new QtCharts::QLineSeries( );
    m_plotData_L2->append(0,0);
    chart->addSeries( m_plotData_classification );
    chart->addSeries( m_plotData_L2 );
    m_XAxis = new QtCharts::QValueAxis();
    m_XAxis->setTitleText("Epoch");
    m_XAxis->setLabelFormat("%d"); 
    m_XAxis->setRange(0,10);
    chart->addAxis(m_XAxis, Qt::AlignBottom);
    m_YAxis = new QtCharts::QValueAxis();
    m_YAxis->setTitleText("Success rate");
    m_YAxis->setRange(0, 100);
    chart->addAxis(m_YAxis, Qt::AlignLeft);
    m_plotData_classification->attachAxis(m_XAxis);
    m_plotData_classification->attachAxis(m_YAxis);
    m_plotData_L2->attachAxis(m_XAxis);
    m_plotData_L2->attachAxis(m_YAxis);
    ui->progressChart->setChart( chart );
    ui->progressChart->setRenderHint(QPainter::Antialiasing);

    prepareSamples();

    // prepare network
    std::vector<unsigned int> map = {784,30,10};
    m_net.reset( new Network(map) );
    m_net->setObserver( this );
    m_net_testing.reset( new Network( *(m_net.get())) );


    m_currentIdx = 0;
    displayTestMNISTImage( m_currentIdx );

    connect( ui->formerSample, &QPushButton::pressed, [=]( )
    {
        if( m_currentIdx == 0 )
            m_currentIdx = m_data->getNumberOfTestSamples() - 1;
        else
            m_currentIdx--;

        displayTestMNISTImage( m_currentIdx );
    });

    connect( ui->nextSample, &QPushButton::pressed, [=]( )
    {
        if( m_currentIdx == m_data->getNumberOfTestSamples() - 1 )
            m_currentIdx = 0;
        else
            m_currentIdx++;

        displayTestMNISTImage( m_currentIdx );
    });


    connect( ui->learnPB, &QPushButton::pressed, [=]( )
    {
        doNNLearning();
    });

    connect( ui->failedSampleList, &QListWidget::itemSelectionChanged, [=]( )
    {
        int currentitem = ui->failedSampleList->currentRow();
        if( currentitem >= 0 && currentitem < m_failedSamples.size() )
        {
            size_t failedIdx = m_failedSamples.at( currentitem );
            displayTestMNISTImage(failedIdx);
        }
    });

    connect( ui->softmax, &QCheckBox::toggled, [=]()
    {
        m_net->setSoftmaxOutput( ui->softmax->isChecked() );
    });

    connect( this, SIGNAL(readyForTesting()), this, SLOT(doNNTesting()));
    connect( this, SIGNAL(readyForLearning()), this, SLOT(doNNLearning()));
    connect( ui->loadNNBtn, SIGNAL(pressed()), this, SLOT(loadNN()));
    connect( ui->saveNNBtn, SIGNAL(pressed()), this, SLOT(saveNN()));


    m_uiUpdaterTimer = new QTimer( this );
    connect(m_uiUpdaterTimer, SIGNAL(timeout()), this, SLOT(updateUi()));
    m_uiUpdaterTimer->start( 100 );
}

Widget::~Widget()
{
    // Note: Since smartpointers are used, objects get deleted automatically.
    delete ui;

    delete m_data;
}

void Widget::prepareSamples()
{
    m_data = new MnistDataInput();

    m_batchin = DataInput::getInputData(m_data->m_training);
    m_batchout = DataInput::getOutputData(m_data->m_training);

    m_testin = DataInput::getInputData(m_data->m_test);
    m_testout = DataInput::getOutputData(m_data->m_test);

    // print lables
    std::cout << "Lables: " << std::endl;
    for( auto dl : m_data->m_lables )
    {
        std::cout << dl.second.lable << " ->  [" << dl.second.output.transpose() << "]" << std::endl;
    }
}

void Widget::displayTestMNISTImage( const size_t& idx )
{
    DataElement sample = m_data->getTestImageAsPixelValues(idx);

    int img_size = 28;
    QImage img(img_size, img_size, QImage::Format_RGB32);
    for( int h = 0; h < img_size; h++ )
    {
        for( int w = 0; w < img_size; w++ )
        {
            uint8_t pixValue = sample.input(28*h+w,0);
            img.setPixel(w, h, qRgb(pixValue, pixValue, pixValue));
        }
    }

    ui->imgLable->setPixmap( QPixmap::fromImage(img.scaled(100,100)) );
    ui->imgLable->show();

    ui->trainingLable->setText( QString::number(sample.lable, 10) );

    if( !m_net_testing->isOperationInProgress() )
    {
        // feedforward
        m_net_testing->feedForward(m_data->m_test.at(idx).input);
        Eigen::MatrixXd activationSignal = m_net_testing->getOutputActivation();
        QString actStr;
        actStr.sprintf("Activation: [ %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]", activationSignal(0,0), activationSignal(1,0),
                       activationSignal(2,0), activationSignal(3,0), activationSignal(4,0), activationSignal(5,0), activationSignal(6,0),
                       activationSignal(7,0), activationSignal(8,0), activationSignal(9,0));
        ui->activationLable->setText(actStr);

        unsigned long maxM, maxN; double maxVal;
        Helpers::maxElement(activationSignal, maxM, maxN, maxVal);
        QString classificationStr; classificationStr.sprintf("Classification: %lu", maxM);
        ui->classificationLable->setText( classificationStr );
    }
}

void Widget::updateUi()
{
    QString testingRes; testingRes.sprintf("Test result L2 = %.2f%%, MaxIdx = %.2f%%", m_sr_L2*100.0, m_sr_MAX * 100.0 );
    ui->resultLable->setText(testingRes);

    ui->learingProgress->setValue( int(round(m_progress_learning * 100.0)) );
    ui->testingProgress->setValue( int(round(m_progress_testing * 100.0)) );

    QMutexLocker locker( &m_uiLock );

    int currentSelectedRow = ui->failedSampleList->currentRow();
    ui->failedSampleList->clear();
    for( size_t i : m_failedSamples )
    {
        QString str; str.sprintf("idx = %lu", i );
        ui->failedSampleList->addItem( str );
    }
    if( currentSelectedRow >= 0 && currentSelectedRow < m_failedSamples.size() )
        ui->failedSampleList->setCurrentRow(currentSelectedRow);

    double current_X_AxisMax = m_XAxis->max();
    if( m_plotData_classification->count() >= current_X_AxisMax )
    {
        m_XAxis->setRange(current_X_AxisMax-10,current_X_AxisMax+10);

        // value axis scaling min max
        // axis scaling : 10 values
        double min_Class; double max_Class;
        getMinMaxYValue(m_plotData_classification,4,min_Class,max_Class);
        double min_L2; double max_L2;
        getMinMaxYValue(m_plotData_L2,4,min_L2,max_L2);

        double lower = std::max(std::min(min_Class,min_L2)*0.95, 0.0);
        double upper = std::min(std::max(max_Class,max_L2)*1.2, 100.0);
        m_YAxis->setRange(lower,upper);
    }
}

void Widget::doNNLearning()
{ 
    double learningRate = ui->learingRateSB->value();
    m_net->setCostFunction( getCurrentSelectedCostFunction() );
    m_net->stochasticGradientDescentAsync(m_batchin, m_batchout, 10, learningRate );
}

void Widget::doNNTesting()
{
    m_net_testing->testNetworkAsync( m_testin, m_testout, 0.50 );
}

void Widget::networkOperationProgress( const NetworkOperationId & opId, const NetworkOperationStatus &opStatus,
                                       const double &progress )
{
    if( opId == NetworkOperationCallback::OpStochasticGradientDescent )
    {
        m_progress_learning = progress;
        if( opStatus == NetworkOperationCallback::OpResultOk )
        {
            // only overwrite if no operation ongoing on testing net
            if( ! m_net_testing->isOperationInProgress() )
            {
                m_net_testing.reset( new Network( *(m_net.get())) );
                emit readyForTesting();
            }

            if( ui->keepLearingCB->isChecked() )
                emit readyForLearning();
        }
    }
    else if( opId == NetworkOperationCallback::OpTestNetwork )
    {
        m_progress_testing = progress;
    }
}

void Widget::networkTestResults( const double& successRateEuclidean, const double& successRateMaxIdx,
                                 const std::vector<std::size_t>& failedSamplesIdx )
{
    QMutexLocker locker( &m_uiLock );

    m_sr_L2 = successRateEuclidean;
    m_sr_MAX = successRateMaxIdx;

    m_failedSamples = failedSamplesIdx;

    m_plotData_L2->append( m_plotData_L2->count(), successRateEuclidean * 100);
    m_plotData_classification->append( m_plotData_classification->count(), successRateMaxIdx * 100 );

    std::cout << "L2 = " << m_sr_L2*100.0 << "%,  MAX = " << m_sr_MAX * 100.0 << "%" << std::endl;
}

void Widget::loadNN()
{
    QString path = QFileDialog::getOpenFileName(this, "Open neuronal network");
    if( path.compare("") != 0 )
    {
        m_net.reset( Network::load( path.toStdString() ) );
        m_net->setObserver( this );
        m_net->setCostFunction( Network::CrossEntropy );

        ui->softmax->setChecked(m_net->isSoftmaxOutputEnabled());

        m_net_testing.reset( new Network( *(m_net.get())) );
        emit readyForTesting();
    }
}

void Widget::saveNN()
{
    QString path = QFileDialog::getSaveFileName(this, "Save neuronal network");
    if( path.compare("") != 0 )
        m_net->save( path.toStdString() );
}

Network::ECostFunction Widget::getCurrentSelectedCostFunction()
{
    int selectedIdx = ui->costFunctionCombo->currentIndex();
    if( selectedIdx == 0 )
    {
        return Network::Quadratic;
    }
    else
    {
        return Network::CrossEntropy;
    }
}

void Widget::getMinMaxYValue(const QtCharts::QLineSeries* series, const uint& nbrEntries, double& min, double& max)
{
    min = std::numeric_limits<double>::max();
    max = std::numeric_limits<double>::min();
    for( int i = series->count()-nbrEntries; i < series->count(); i++ )
    {
        if( i >= 0 )
        {
            if( series->at(i).y() < min )
            {
                min = series->at(i).y();
            }

            if( series->at(i).y() > max )
            {
                max = series->at(i).y();
            }
        }
    }
}


