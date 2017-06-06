#include "widget.h"
#include "ui_widget.h"
#include "layer.h"
#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

#include "QPixmap"
#include "QImage"


Widget::Widget(QWidget* parent) : QMainWindow(parent), ui(new Ui::Widget)
{
    ui->setupUi(this);

    // load mnist data set
    prepareSamples();
    m_currentIdx = 0;
    displayMNISTImage( m_currentIdx );

    connect( ui->formerSample, &QPushButton::pressed, [=]( )
    {
        if( m_currentIdx == 0 )
            m_currentIdx = m_trainingSet.size() - 1;
        else
            m_currentIdx--;

        displayMNISTImage( m_currentIdx );
    });

    connect( ui->nextSample, &QPushButton::pressed, [=]( )
    {
        if( m_currentIdx == m_trainingSet.size() - 1 )
            m_currentIdx = 0;
        else
            m_currentIdx++;

        displayMNISTImage( m_currentIdx );
    });

    learn();
    //sameImage();

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

void Widget::displayMNISTImage( const size_t& idx )
{
    NNSample sample = m_trainingSet.at(idx);

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
}

void Widget::sameImage()
{
    std::vector<unsigned int> map = {784,100,10};
    Network* net = new Network(map);

    NNSample sample = m_trainingSet.at(0);

    for( int k = 0; k < 10000; k++ )
    {
        net->gradientDescent(sample.normalizedinput,sample.output,3.0);
    }

    Eigen::MatrixXd err = net->getOutputLayer()->getBackpropagationError();
    std::cout << "Error = " << err.norm() << ": " << err.transpose() << std::endl;

    net->feedForward(sample.normalizedinput);
    Eigen::MatrixXd outSignal = net->getOutputActivation();

    std::cout << "Outsignal = " << outSignal.transpose() << std::endl;
    std::cout << "Lable = " << int(sample.lable)  << " : " << sample.output.transpose() << std::endl;

    delete net;
}

void Widget::learn()
{
    std::vector<unsigned int> map = {784,40,10};
    Network* net = new Network(map);
    unsigned int nbrEpochs = 1000;

    // prepare batch
    std::vector<Eigen::MatrixXd> batchin;
    std::vector<Eigen::MatrixXd> batchout;
    for( size_t z = 0; z < m_trainingSet.size(); z++ )
    {
        batchout.push_back( m_trainingSet.at(z).output );
        batchin.push_back( m_trainingSet.at(z).normalizedinput );
    }

    unsigned int batchsize = 10;

    for( unsigned int epoch = 0; epoch < nbrEpochs; epoch++ )
    {
        // training
        net->stochasticGradientDescent(batchin, batchout, batchsize, 3.0 );

        // test
        double successfull = 0;
        for( size_t t = 0; t < m_testingSet.size(); t++ )
        {
            NNSample sample = m_testingSet.at(t);

            net->feedForward(sample.normalizedinput);
            Eigen::MatrixXd out = net->getOutputActivation();
            double maxElement = 0.0;
            int maxIdx = 0;
            for( int j = 0; j < out.rows(); j++ )
            {
                if( maxElement < out(j,0) )
                {
                    maxElement =  out(j,0);
                    maxIdx = j;
                }
            }

            if( maxIdx == sample.lable )
                successfull = successfull + 1;
            else
                maxIdx = 0;

        }

        std::cout << "Epoch " << epoch <<  ": success rate = " << 100 * successfull / double(m_testingSet.size()) << "% , bkError: " << net->getOutputLayer()->getBackpropagationError().norm() << std::endl;
    }

    delete net;
}


