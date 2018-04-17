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
    m_sr_L2(0), m_sr_MAX(0), m_progress_testing(0), m_progress_learning(0), m_networkError(0), m_cam(NULL)
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
    prepareSamplesAndNetwork();


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

    connect( ui->startLiveBtn, &QPushButton::pressed, [=]( )
    {
        if( m_processLiveTimer->isActive() )
        {
            m_processLiveTimer->stop();
            m_uiUpdaterTimer->start( 250 );
        }
        else
        {
            // init camera, cascade filters and start recording
            if( !m_cam )
            {
                m_cam = new cv::VideoCapture(0);

                if( !m_face_cascade.load("/Users/eidelen/dev/libs/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_default.xml"))
                {
                    std::cerr << "Could not open face cascade filter" << std::endl;
                    return;
                }

                if( !m_eyes_cascade.load("/Users/eidelen/dev/libs/opencv-3.2.0/data/haarcascades/haarcascade_eye.xml"))
                {
                    std::cerr << "Could not open face cascade filter" << std::endl;
                    return;
                }
            }

            m_uiUpdaterTimer->stop();
            m_processLiveTimer->start(100);
        }
    });

    m_processLiveTimer = new QTimer( this );
    connect(m_processLiveTimer, SIGNAL(timeout()), this, SLOT(doLive()));


    m_uiUpdaterTimer = new QTimer( this );
    connect(m_uiUpdaterTimer, SIGNAL(timeout()), this, SLOT(updateUi()));
    m_uiUpdaterTimer->start( 100 );
}

Widget::~Widget()
{
    // Note: Since smartpointers are used, objects get deleted automatically.
    delete ui;

    if(m_cam)
    {
        m_cam->release();
        delete m_cam;
    }
}

void Widget::prepareSamplesAndNetwork()
{
    m_data = new FaceDataInput();

    m_data->addToTraining("data/adrian.csv", 0);
    m_data->addToTraining("data/adrian_glasses.csv", 1);

    m_data->addToTest("data/adrian_test.csv",0);
    m_data->addToTest("data/adrian_glasses_test.csv",1);

    m_testDataVisible = DataInput::getInputData(m_data->m_test); // unnormalized data -> call before normalizeData()

    m_data->generateFromLables();
    m_data->normalizeData(); // very important

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

    std::cout << "Number of training samples = " << m_data->getNumberOfTrainingSamples() << std::endl;
    std::cout << "Number of test samples = " << m_data->getNumberOfTestSamples() << std::endl;

    // validate sample data
    DataInput::DataInputValidation div = m_data->validateData();

    if( div.valid )
    {
        // prepare network
        std::vector<unsigned int> map;
        map.push_back(div.inputDataLength);
        map.push_back(1024);
        map.push_back(128);
        map.push_back(div.outputDataLength);

        m_net.reset(new Network(map));
        m_net->setObserver(this);
        m_net_testing.reset(new Network(*(m_net.get())));
    }
    else
    {
        std::cerr << "Invalid sample data" << std::endl;
        std::exit(-1);
    }
}

Eigen::MatrixXd Widget::lableToOutputVector( const uint8_t& lable )
{
    Eigen::MatrixXd ret = Eigen::MatrixXd::Constant(2,1, 0.0);
    ret( lable, 0 ) = 1.0;
    return ret;
}

void Widget::drawImg(const Eigen::MatrixXd& imgdata)
{
    int img_size = 64;
    QImage img(img_size, img_size, QImage::Format_RGB32);
    for( int h = 0; h < img_size; h++ )
    {
        for( int w = 0; w < img_size; w++ )
        {
            uint8_t pixValue = uint8_t( round( imgdata(64*h+w,0)  ) );
            img.setPixel(w, h, qRgb(pixValue, pixValue, pixValue));
        }
    }

    ui->imgLable->setPixmap( QPixmap::fromImage(img.scaled(100,100)) );
    ui->imgLable->show();

}

void Widget::displayTestMNISTImage( const size_t& idx )
{
    Eigen::MatrixXd visImg = m_testDataVisible.at(idx);
    Eigen::MatrixXd lable = m_testout.at(idx);

    drawImg(visImg);

    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

    std::ostringstream lableStr;
    lableStr << "Lable: " << lable.transpose().format(CleanFmt);
    ui->classificationLable->setText(QString(lableStr.str().c_str()));

    if( !m_net_testing->isOperationInProgress() )
    {
        // feedforward
        m_net_testing->feedForward(m_testin.at(idx));
        Eigen::MatrixXd activationSignal = m_net_testing->getOutputActivation();

        std::ostringstream signalStr;
        signalStr << "Activation: " << activationSignal.transpose().format(CleanFmt);

        ui->activationLable->setText(QString(signalStr.str().c_str()));
    }
}

void Widget::updateUi()
{
    QString testingRes; testingRes.sprintf("Test result L2 = %.2f%%, MaxIdx = %.2f%%", m_sr_L2*100.0, m_sr_MAX * 100.0 );
    ui->resultLable->setText(testingRes);

    ui->learingProgress->setValue( int(round(m_progress_learning * 100.0)) );
    ui->testingProgress->setValue( int(round(m_progress_testing * 100.0)) );

    QMutexLocker locker( &m_uiLock );

    QString errStr( "Error: ");
    errStr.append( QString::number(m_networkError) );
    ui->errLable->setText(errStr);

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
        m_networkError = m_net->getNetworkErrorMagnitude();

        if( opStatus == NetworkOperationCallback::OpResultOk )
        {
            // only overwrite if no operation ongoing on testing net
            if( ! m_net_testing->isOperationInProgress() )
            {
                QMutexLocker locker( &m_uiLock );
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


void Widget::doLive()
{
    if( !m_cam->isOpened() )
    {
        std::cerr << "Camera stream not open" << std::endl;
        return;
    }

    cv::Mat image;
    cv::Mat grayImg;
    (*m_cam) >> image;
    cv::cvtColor(image, grayImg, cv::COLOR_BGR2GRAY);

    std::vector<cv::Rect> faces;
    m_face_cascade.detectMultiScale(grayImg,faces,1.1,5);

    for( const cv::Rect& f : faces )
    {
        double x = f.x;
        double y = f.y;

        cv::Mat faceROI = grayImg( f );
        std::vector<cv::Rect> eyes;
        m_eyes_cascade.detectMultiScale(faceROI, eyes);

        // two eyes
        if( eyes.size() == 2 )
        {
            // crop image region
            double e1x = eyes.at(0).x;
            double e1y = eyes.at(0).y;
            double e1w = eyes.at(0).width;
            double e1h = eyes.at(0).height;

            double e2x = eyes.at(1).x;
            double e2y = eyes.at(1).y;
            double e2w = eyes.at(1).width;
            double e2h = eyes.at(1).height;


            double eye_dist = std::abs((e1x+(e1w/2.0) - (e2x+(e2w/2.0))));

            double cut_cent_x = (e1x+(e1w/2.0) + e2x+(e2w/2.0)) / 2.0;
            double cut_cent_y = (e1y+(e1h/2.0) + e2y+(e2h/2.0)) / 2.0;

            int region_x = int(std::round(cut_cent_x - 0.8 * eye_dist));
            int region_w = int(std::round(1.6 * eye_dist));

            int region_y = int(std::round(cut_cent_y - eye_dist*0.2));
            int region_h = region_w;

            cv::Rect vipRect(region_x, region_y, region_w, region_h);

            // check if valid rect
            if( (vipRect & cv::Rect(0, 0, faceROI.cols, faceROI.rows)) != vipRect )
            {
                std::cerr << "Wrong rect." << std::endl;
                return;
            }

            cv::Mat vip_region = faceROI(vipRect);

            // resize
            cv::Mat nn_input;
            cv::resize(vip_region,nn_input,cv::Size(64,64),0,0,cv::INTER_CUBIC);
            Eigen::MatrixXd inVector(64*64,1);
            for(int m=0; m<nn_input.rows; m++)
            {
                for (int n = 0; n < nn_input.cols; n++)
                {
                    inVector(m*64 + n) = nn_input.at<unsigned char>(m,n);
                }
            }

            drawImg(inVector);

            Eigen::MatrixXd inNorm = DataInput::normalize0Mean1Std(inVector);

            // classify in neuronal network
            m_net_testing->feedForward(inNorm);
            Eigen::MatrixXd activationSignal = m_net_testing->getOutputActivation();

            Eigen::IOFormat CleanFmt(8, 0, ", ", "\n", "[", "]");
            std::ostringstream signalStr;

            int lable = DataInput::getStrongestIdx(activationSignal);

            signalStr << "Activation: " << lable << "    : " << activationSignal.transpose().format(CleanFmt);

            ui->activationLable->setText(QString(signalStr.str().c_str()));


            cv::rectangle(image, cv::Point(x+region_x, y+region_y), cv::Point(x+region_x+region_w, y+region_y+region_h), cv::Scalar(0, 255, 0), 4);
            cv::rectangle(image, cv::Point(x+e1x, y+e1y), cv::Point(x+e1x + e1w, y+e1y + e1h), cv::Scalar(0, 0, 255), 2);
            cv::rectangle(image, cv::Point(x+e2x, y + e2y), cv::Point(x + e2x + e2w, y + e2y + e2h), cv::Scalar(0, 0, 255), 2);

            std::ostringstream imgText;
            imgText << "Lable = " << lable;
            cv::putText(image, imgText.str(), cv::Point(10,100), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 2);
        }
    }

    cv::imshow("facedetection", image);
}


