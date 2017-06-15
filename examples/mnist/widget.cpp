#include "widget.h"
#include "ui_widget.h"
#include "layer.h"
#include "helpers.h"
#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

#include <QPixmap>
#include <QImage>
#include <QMutexLocker>
#include <QStringListModel>
#include <QChart>
#include <QChartView>

Widget::Widget(QWidget* parent) : QMainWindow(parent), ui(new Ui::Widget),
    m_sr_L2(0), m_sr_MAX(0), m_progress(0)
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
    chart->addAxis(m_XAxis, Qt::AlignBottom);
    QtCharts::QValueAxis* yAxis = new QtCharts::QValueAxis();
    yAxis->setTitleText("Success");
    yAxis->setRange(0, 100);
    chart->addAxis(yAxis, Qt::AlignLeft);
    m_plotData_classification->attachAxis(m_XAxis);
    m_plotData_classification->attachAxis(yAxis);
    m_plotData_L2->attachAxis(m_XAxis);
    m_plotData_L2->attachAxis(yAxis);
    ui->progressChart->setChart( chart );
    ui->progressChart->setRenderHint(QPainter::Antialiasing);

    // load mnist data set
    prepareSamples();

    // prepare network
    std::vector<unsigned int> map = {784,30,10};
    m_net.reset( new Network(map) );
    m_net->setObserver( this );


    m_currentIdx = 0;
    displayTestMNISTImage( m_currentIdx );

    connect( ui->formerSample, &QPushButton::pressed, [=]( )
    {
        if( m_currentIdx == 0 )
            m_currentIdx = m_testingSet.size() - 1;
        else
            m_currentIdx--;

        displayTestMNISTImage( m_currentIdx );
    });

    connect( ui->nextSample, &QPushButton::pressed, [=]( )
    {
        if( m_currentIdx == m_testingSet.size() - 1 )
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

    connect( this, SIGNAL(readyForTesting()), this, SLOT(doNNTesting()));
    connect( this, SIGNAL(readyForLearning()), this, SLOT(doNNLearning()));


    m_uiUpdaterTimer = new QTimer( this );
    connect(m_uiUpdaterTimer, SIGNAL(timeout()), this, SLOT(updateUi()));
    m_uiUpdaterTimer->start( 100 );
}

Widget::~Widget()
{
    // Note: Since smartpointers are used, objects get deleted automatically.
    delete ui;
}

void Widget::prepareSamples()
{
    auto mnistinput = mnist::read_dataset<std::vector, std::vector, double, uint8_t>();
    auto mnistinputNormalized = mnist::read_dataset<std::vector, std::vector, double, uint8_t>();
    mnist::normalize_dataset(mnistinputNormalized);

    // Load training data
    m_trainingSet.clear();
    for( size_t k = 0; k < mnistinput.training_images.size(); k++ )
    {
        Eigen::MatrixXd xIn; Eigen::MatrixXd xInNormalized; uint8_t lable;
        loadMNISTSample( mnistinput.training_images, mnistinput.training_labels, k, xIn, lable );
        loadMNISTSample( mnistinputNormalized.training_images, mnistinputNormalized.training_labels, k, xInNormalized, lable );
        Eigen::MatrixXd yOut = lableToOutputVector( lable );

        NNSample thisSample;
        thisSample.input = xIn; thisSample.normalizedinput = xInNormalized;
        thisSample.output = yOut; thisSample.lable = lable;
        m_trainingSet.push_back( thisSample );
    }

    // Load testing data
    m_testingSet.clear();
    for( size_t k = 0; k < mnistinput.test_images.size(); k++ )
    {
        Eigen::MatrixXd xIn; Eigen::MatrixXd xInNormalized; uint8_t lable;
        loadMNISTSample( mnistinput.test_images, mnistinput.test_labels, k, xIn, lable );
        loadMNISTSample( mnistinputNormalized.test_images, mnistinputNormalized.test_labels, k, xInNormalized, lable );
        Eigen::MatrixXd yOut = lableToOutputVector( lable );

        NNSample thisSample;
        thisSample.input = xIn; thisSample.normalizedinput = xInNormalized;
        thisSample.output = yOut; thisSample.lable = lable;
        m_testingSet.push_back( thisSample );
    }

    // Prepare batchs
    m_batchin.clear(); m_batchout.clear();
    for( size_t z = 0; z < m_trainingSet.size(); z++ )
    {
        m_batchout.push_back( m_trainingSet.at(z).output );
        m_batchin.push_back( m_trainingSet.at(z).normalizedinput );
    }

    m_testin.clear(); m_testout.clear();
    for( size_t z = 0; z < m_testingSet.size(); z++ )
    {
        m_testout.push_back( m_testingSet.at(z).output );
        m_testin.push_back( m_testingSet.at(z).normalizedinput );
    }
}

bool Widget::loadMNISTSample( const std::vector<std::vector<double>>& imgSet, const std::vector<uint8_t>& lableSet,
                              const size_t& idx, Eigen::MatrixXd& img, uint8_t& lable)
{
    if( max(imgSet.size(), lableSet.size()) <= idx )
        return false;

    lable = lableSet.at( idx );
    const std::vector<double>& imgV = imgSet.at( idx );

    img = Eigen::MatrixXd( imgV.size(), 1 );
    for( size_t i = 0; i < imgV.size(); i++ )
        img( int(i), 0 ) = imgV.at(i);

    return true;
}

Eigen::MatrixXd Widget::lableToOutputVector( const uint8_t& lable )
{
    Eigen::MatrixXd ret = Eigen::MatrixXd::Constant(10,1, 0.0);
    ret( lable, 0 ) = 1.0;
    return ret;
}

void Widget::displayTestMNISTImage( const size_t& idx )
{
    NNSample sample = m_testingSet.at(idx);

    int img_size = 28;
    QImage img(img_size, img_size, QImage::Format_RGB32);
    for( int h = 0; h < img_size; h++ )
    {
        for( int w = 0; w < img_size; w++ )
        {
            uint8_t pixValue = uint8_t( round( sample.input(28*h+w,0) ) );
            img.setPixel(w, h, qRgb(pixValue, pixValue, pixValue));
        }
    }

    ui->imgLable->setPixmap( QPixmap::fromImage(img.scaled(140,140)) );
    ui->imgLable->show();

    ui->trainingLable->setText( QString::number(sample.lable, 10) );

    if( !m_net->isOperationInProgress() )
    {
        // feedforward
        m_net->feedForward(sample.normalizedinput);
        Eigen::MatrixXd activationSignal = m_net->getOutputActivation();
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

    ui->operationProgressBar->setValue( int(round(m_progress * 100.0)) );

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

    m_XAxis->setRange(0,m_plotData_classification->count()+1);
}

void Widget::doNNLearning()
{ 
    ui->operationLable->setText("SGD learning...");
    m_net->stochasticGradientDescentAsync(m_batchin, m_batchout, 10, 3.0 );
}

void Widget::doNNTesting()
{
    ui->operationLable->setText("Network testing...");
    m_net->testNetworkAsync( m_testin, m_testout, 0.50 );
}

void Widget::networkOperationProgress( const NetworkOperationId & opId, const NetworkOperationStatus &opStatus,
                                       const double &progress )
{
    m_progress = progress;

    if( opId == NetworkOperationCallback::OpStochasticGradientDescent )
    {
        if( opStatus == NetworkOperationCallback::OpResultOk )
            emit readyForTesting();
    }
    else if( opId == NetworkOperationCallback::OpTestNetwork )
    {}
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

    if( ui->keepLearingCB->isChecked() )
        emit readyForLearning();
}


