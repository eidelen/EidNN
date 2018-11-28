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
    m_sr_L2(0), m_sr_MAX(0), m_progress_training_testing(0), m_progress_learning(0),
    m_progress_validation(0)
{
    ui->setupUi(this);

    QtCharts::QChart* trainingCostChart = new QtCharts::QChart( );
    trainingCostChart->legend()->hide();
    m_trainingSetCost = new QtCharts::QLineSeries( );
    trainingCostChart->addSeries( m_trainingSetCost );
    m_RCXAxis = new QtCharts::QValueAxis();
    m_RCXAxis->setTitleText("Epoch");
    m_RCXAxis->setLabelFormat("%d");
    m_RCXAxis->setRange(0,1);
    trainingCostChart->addAxis(m_RCXAxis, Qt::AlignBottom);
    m_RCYAxis = new QtCharts::QValueAxis();
    m_RCYAxis->setTitleText("Cost");
    m_RCYAxis->setRange(0, 0.1);
    trainingCostChart->addAxis(m_RCYAxis, Qt::AlignLeft);
    m_trainingSetCost->attachAxis(m_RCXAxis);
    m_trainingSetCost->attachAxis(m_RCYAxis);
    ui->trainingerror_chart->setChart(trainingCostChart);
    ui->trainingerror_chart->setRenderHint(QPainter::Antialiasing);

    // create chart
    QtCharts::QChart *chart = new QtCharts::QChart( );
    chart->legend()->hide();
    m_plotData_classification = new QtCharts::QLineSeries( );
    m_plotData_classification->append(0,0);
    m_trainingSuccess = new QtCharts::QLineSeries( );
    m_trainingSuccess->append(0,0);
    chart->addSeries( m_plotData_classification );
    chart->addSeries( m_trainingSuccess );
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
    m_trainingSuccess->attachAxis(m_XAxis);
    m_trainingSuccess->attachAxis(m_YAxis);
    ui->progressChart->setChart( chart );
    ui->progressChart->setRenderHint(QPainter::Antialiasing);

    QtCharts::QChart* testCostChart = new QtCharts::QChart( );
    testCostChart->legend()->hide();
    m_testSetCost = new QtCharts::QLineSeries( );
    testCostChart->addSeries( m_testSetCost );
    m_TCXAxis = new QtCharts::QValueAxis();
    m_TCXAxis->setTitleText("Epoch");
    m_TCXAxis->setLabelFormat("%d");
    m_TCXAxis->setRange(0,1);
    testCostChart->addAxis(m_TCXAxis, Qt::AlignBottom);
    m_TCYAxis = new QtCharts::QValueAxis();
    m_TCYAxis->setTitleText("Cost");
    m_TCYAxis->setRange(0, 0.1);
    testCostChart->addAxis(m_TCYAxis, Qt::AlignLeft);
    m_testSetCost->attachAxis(m_TCXAxis);
    m_testSetCost->attachAxis(m_TCYAxis);
    ui->testerror_chart->setChart(testCostChart);
    ui->testerror_chart->setRenderHint(QPainter::Antialiasing);

    prepareSamples();

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

    connect( ui->regCB, &QCheckBox::toggled, [=]()
    {
        ui->regLambdaSB->setEnabled(ui->regCB->isChecked());
        setRegularizationFunction();
    });

    connect( ui->regLambdaSB, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), [=]()
    {
        setRegularizationFunction();
    });

    connect( ui->resetBtn, &QPushButton::pressed, [=]( )
    {
        if( !m_net->isOperationInProgress() )
        {
            m_net->resetWeights();
        }
    });

    connect( this, SIGNAL(readyForValidation()), this, SLOT(doNNValidation()));
    connect( this, SIGNAL(readyForTrainingTesting()), this, SLOT(doNNTesting()));
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

    // validate sample data
    DataInput::DataInputValidation div = m_data->validateData();

    if( div.valid )
    {
        // prepare network
        std::vector<unsigned int> map;
        map.push_back(div.inputDataLength);
        map.push_back(100);
        map.push_back(div.outputDataLength);

        m_net.reset(new Network(map));
        m_net->setObserver(this);
        m_net_validation.reset(new Network(*(m_net.get())));
        m_net_training_testing.reset(new Network(*(m_net.get())));
    }
    else
    {
            std::cerr << "Invalid sample data" << std::endl;
            std::exit(-1);
    }
}

void Widget::displayTestMNISTImage( const size_t& idx )
{
    DataElement sample = m_data->getTestImageAsPixelValues(idx);
    bool repAvailable = false;
    Eigen::MatrixXd imgMatrix = m_data->representation(sample.input, &repAvailable);

    if( repAvailable )
    {
        QImage img(imgMatrix.cols(), imgMatrix.rows(), QImage::Format_RGB32);
        for (int h = 0; h < imgMatrix.rows(); h++)
        {
            for (int w = 0; w < imgMatrix.cols(); w++)
            {
                uint8_t pixValue = imgMatrix(h,w);
                img.setPixel(w, h, qRgb(pixValue, pixValue, pixValue));
            }
        }

        ui->imgLable->setPixmap(QPixmap::fromImage(img.scaled(100, 100)));
        ui->imgLable->show();
    }

    ui->testlable->setText( "Lable: " + QString::number(sample.lable, 10) );

    if( !m_net_validation->isOperationInProgress() )
    {
        // feedforward
        m_net_validation->feedForward(m_data->m_test.at(idx).input);
        Eigen::MatrixXd activationSignal = m_net_validation->getOutputActivation();
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
    ui->testingTrainingProgress->setValue( int(round(m_progress_training_testing * 100.0)) );
    ui->validationProgress->setValue( int(round(m_progress_validation * 100.0)) );

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
        getMinMaxYValue(m_trainingSuccess,4,min_L2,max_L2);

        double lower = std::max(std::min(min_Class,min_L2)*0.95, 0.0);
        double upper = std::min(std::max(max_Class,max_L2)*1.2, 100.0);
        m_YAxis->setRange(lower,upper);
    }

    if( m_testSetCost->count() > 0 )
    {
        m_TCXAxis->setRange(0, m_testSetCost->count());
        double min_TYVal;
        double max_TYVal;
        getMinMaxYValue(m_testSetCost, m_testSetCost->count(), min_TYVal, max_TYVal);
        m_TCYAxis->setRange(0, max_TYVal);
    }

    if( m_trainingSetCost->count() > 0 )
    {
        m_RCXAxis->setRange(0, m_trainingSetCost->count());
        double min_RYVal;
        double max_RYVal;
        getMinMaxYValue(m_trainingSetCost, m_trainingSetCost->count(), min_RYVal, max_RYVal);
        m_RCYAxis->setRange(0, max_RYVal);
    }

    ui->resetBtn->setEnabled( !m_net->isOperationInProgress()) ;
}

void Widget::doNNLearning()
{ 
    double learningRate = ui->learingRateSB->value();
    m_net->setCostFunction( getCurrentSelectedCostFunction() );
    m_net->stochasticGradientDescentAsync(m_batchin, m_batchout, 10, learningRate, NETID_TRAINING );
}

void Widget::doNNTesting()
{
    m_net_training_testing->testNetworkAsync( m_batchin, m_batchout, 0.50, NETID_TRAINING_TESTING);
}

void Widget::doNNValidation()
{
    m_net_validation->testNetworkAsync( m_testin, m_testout, 0.50, NETID_VALIDATION);
}

void Widget::networkOperationProgress( const NetworkOperationId & opId, const NetworkOperationStatus &opStatus,
                                       const double &progress, const int& userId )
{
    if( opId == NetworkOperationCallback::OpStochasticGradientDescent )
    {
        m_progress_learning = progress;
        if( opStatus == NetworkOperationCallback::OpResultOk )
        {
            // only overwrite if no operation ongoing on validation net
            if( ! m_net_validation->isOperationInProgress() )
            {
                m_net_validation.reset( new Network( *(m_net.get())) );
                emit readyForValidation();
            }

            // only overwrite if no operation ongoing on training testing net
            if( ! m_net_training_testing->isOperationInProgress() )
            {
                m_net_training_testing.reset( new Network( *(m_net.get())) );
                emit readyForTrainingTesting();
            }

            if( ui->keepLearingCB->isChecked() )
                emit readyForLearning();
        }
    }
    else if( opId == NetworkOperationCallback::OpTestNetwork )
    {
        if( userId == NETID_VALIDATION )
            m_progress_validation = progress;
        else if( userId == NETID_TRAINING_TESTING )
            m_progress_training_testing = progress;
    }
}

void Widget::networkTestResults( const double& successRateEuclidean, const double& successRateMaxIdx,
                                 const double& averageCost,
                                 const std::vector<std::size_t>& failedSamplesIdx, const int& userId )
{
    QMutexLocker locker( &m_uiLock );

    if( userId == NETID_VALIDATION )
    {
        m_sr_L2 = successRateEuclidean;
        m_sr_MAX = successRateMaxIdx;

        m_failedSamples = failedSamplesIdx;

        m_plotData_classification->append(m_plotData_classification->count(), successRateMaxIdx * 100);

        m_testSetCost->append(m_testSetCost->count(), averageCost);

        std::cout << "Test success: L2 = " << m_sr_L2 * 100.0 << "%,  MAX = " << m_sr_MAX * 100.0 << "%" << std::endl;
        std::cout << "AVG TEST COST = " << averageCost << std::endl;
    }
    else if( userId == NETID_TRAINING_TESTING )
    {
        m_trainingSuccess->append( m_trainingSuccess->count(), successRateMaxIdx * 100);
        m_trainingSetCost->append( m_trainingSetCost->count(), averageCost );

        std::cout << "Training success: L2 = " << successRateEuclidean*100.0 << "%,  MAX = " << successRateMaxIdx * 100.0 << "%" << std::endl;
        std::cout << "AVG TRAINING COST = " << averageCost << std::endl;
    }
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

        m_net_validation.reset( new Network( *(m_net.get())) );
        m_net_training_testing.reset( new Network( *(m_net.get())) );

        emit readyForValidation();
        emit readyForTrainingTesting();
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
        return Network::CrossEntropy;
    }
    else
    {
        return Network::Quadratic;
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

void Widget::setRegularizationFunction()
{
    std::shared_ptr<Regularization> reg(new Regularization(Regularization::RegularizationMethod::NoneRegularization, 1));

    if( ui->regCB->isChecked() )
        reg.reset( new Regularization(Regularization::RegularizationMethod::WeightDecay, ui->regLambdaSB->value()) );

    m_net->setRegularizationMethod(reg);

    std::cout << "Set regularization method: " << reg->toString() << " lambda: " << reg->m_lamda << std::endl;
}


