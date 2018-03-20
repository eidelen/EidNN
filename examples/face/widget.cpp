#include "widget.h"
#include "ui_widget.h"
#include "layer.h"
#include "helpers.h"
#include <limits>
#include <iostream>

#include <QPixmap>
#include <QImage>
#include <QMutexLocker>
#include <QStringListModel>
#include <QChart>
#include <QChartView>
#include <QFileDialog>
#include <QTextStream>

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

    // load mnist data set
    prepareSamples();

    // prepare network
    std::vector<unsigned int> map = {4096,1024,128,2};
    m_net.reset( new Network(map) );
    m_net->setObserver( this );
    m_net_testing.reset( new Network( *(m_net.get())) );


    m_currentIdx = 0;
    displayTestMNISTImage( m_currentIdx );

    connect( ui->formerSample, &QPushButton::pressed, [=]( )
    {
        if( m_currentIdx == 0 )
            m_currentIdx = m_testin.size() - 1;
        else
            m_currentIdx--;

        displayTestMNISTImage( m_currentIdx );
    });

    connect( ui->nextSample, &QPushButton::pressed, [=]( )
    {
        if( m_currentIdx == m_testin.size() - 1 )
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
}

void Widget::prepareSamples()
{
    m_batchin.clear(); m_batchout.clear();
    m_testin.clear(); m_testout.clear();

    {
        QFile inputFileAdrian("data/adrian.csv");
        if (inputFileAdrian.open(QIODevice::ReadOnly))
        {
            int k = 0;
            QTextStream in(&inputFileAdrian);
            while (!in.atEnd())
            {
                Eigen::MatrixXd xInNormalized = Eigen::MatrixXd(4096, 1);

                QString line = in.readLine();
                QStringList values = line.split(",");
                for (int i = 0; i < 4096; i++)
                {
                    QString vStr = values[i];

                    bool okTrans = true;
                    float val = vStr.toFloat(&okTrans);
                    if (!okTrans)
                    {
                        std::cout << "Error to float" << std::endl;
                        return;
                    }

                    xInNormalized(i, 0) = val;
                }

                Eigen::MatrixXd yOut = lableToOutputVector(0);

                if( false ) //k % 5 == 0 ) // every 5th to testing
                {
                    m_testin.push_back(xInNormalized);
                    m_testout.push_back(yOut);
                }
                else
                {
                    m_batchin.push_back(xInNormalized);
                    m_batchout.push_back(yOut);
                }

                k++;

            }
            inputFileAdrian.close();
        }
    }


    {
        QFile inputFileAdrianGlasses("data/adrian_glasses.csv");
        if (inputFileAdrianGlasses.open(QIODevice::ReadOnly))
        {
            int k = 0;
            QTextStream in(&inputFileAdrianGlasses);
            while (!in.atEnd())
            {
                Eigen::MatrixXd xInNormalized = Eigen::MatrixXd(4096, 1);

                QString line = in.readLine();
                QStringList values = line.split(",");
                for (int i = 0; i < 4096; i++)
                {
                    QString vStr = values[i];

                    bool okTrans = true;
                    float val = vStr.toFloat(&okTrans);
                    if (!okTrans)
                    {
                        std::cout << "Error to float" << std::endl;
                        return;
                    }

                    xInNormalized(i, 0) = val;
                }

                Eigen::MatrixXd yOut = lableToOutputVector(1);

                if( false ) //k % 5 == 0 ) // every 5th to testing
                {
                    m_testin.push_back(xInNormalized);
                    m_testout.push_back(yOut);
                }
                else
                {
                    m_batchin.push_back(xInNormalized);
                    m_batchout.push_back(yOut);
                }

                k++;

            }
            inputFileAdrianGlasses.close();
        }
    }


    {
        QFile inputFileAdrianTest("data/adrian_test.csv");
        if (inputFileAdrianTest.open(QIODevice::ReadOnly))
        {
            QTextStream in(&inputFileAdrianTest);
            while (!in.atEnd())
            {
                Eigen::MatrixXd xInNormalized = Eigen::MatrixXd(4096, 1);

                QString line = in.readLine();
                QStringList values = line.split(",");
                for (int i = 0; i < 4096; i++)
                {
                    QString vStr = values[i];

                    bool okTrans = true;
                    float val = vStr.toFloat(&okTrans);
                    if (!okTrans)
                    {
                        std::cout << "Error to float" << std::endl;
                        return;
                    }

                    xInNormalized(i, 0) = val;
                }

                Eigen::MatrixXd yOut = lableToOutputVector(0);

                m_testin.push_back(xInNormalized);
                m_testout.push_back(yOut);
            }
            inputFileAdrianTest.close();

        }
    }

    {
        QFile inputFileAdrianGlassesTest("data/adrian_glasses_test.csv");
        if (inputFileAdrianGlassesTest.open(QIODevice::ReadOnly))
        {
            QTextStream in(&inputFileAdrianGlassesTest);
            while (!in.atEnd())
            {
                Eigen::MatrixXd xInNormalized = Eigen::MatrixXd(4096, 1);

                QString line = in.readLine();
                QStringList values = line.split(",");
                for (int i = 0; i < 4096; i++)
                {
                    QString vStr = values[i];

                    bool okTrans = true;
                    float val = vStr.toFloat(&okTrans);
                    if (!okTrans)
                    {
                        std::cout << "Error to float" << std::endl;
                        return;
                    }

                    xInNormalized(i, 0) = val;
                }

                Eigen::MatrixXd yOut = lableToOutputVector(1);

                m_testin.push_back(xInNormalized);
                m_testout.push_back(yOut);
            }
            inputFileAdrianGlassesTest.close();
        }
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
    Eigen::MatrixXd ret = Eigen::MatrixXd::Constant(2,1, 0.0);
    ret( lable, 0 ) = 1.0;
    return ret;
}

void Widget::displayTestMNISTImage( const size_t& idx )
{
    Eigen::MatrixXd normImg = m_testin.at(idx);
    Eigen::MatrixXd lable = m_testout.at(idx);

    int img_size = 64;
    QImage img(img_size, img_size, QImage::Format_RGB32);
    for( int h = 0; h < img_size; h++ )
    {
        for( int w = 0; w < img_size; w++ )
        {
            uint8_t pixValue = uint8_t( round( (normImg(64*h+w,0) + 1.0)*128 ) );
            img.setPixel(w, h, qRgb(pixValue, pixValue, pixValue));
        }
    }

    ui->imgLable->setPixmap( QPixmap::fromImage(img.scaled(100,100)) );
    ui->imgLable->show();

    if( lable(0,0) > lable(1,0) )
        ui->trainingLable->setText( "No Glasses");
    else
        ui->trainingLable->setText( "Glasses");

    if( !m_net_testing->isOperationInProgress() )
    {
        // feedforward
        m_net_testing->feedForward(normImg);
        Eigen::MatrixXd activationSignal = m_net_testing->getOutputActivation();
        QString actStr;
        actStr.sprintf("Activation: [ %.2f, %.2f ]", activationSignal(0,0), activationSignal(1,0));
        ui->activationLable->setText(actStr);
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
    m_net->stochasticGradientDescentAsync(m_batchin, m_batchout, 16, learningRate );
}

void Widget::doNNTesting()
{
    m_net_testing->testNetworkAsync( m_testin, m_testout, 0.3 );
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


