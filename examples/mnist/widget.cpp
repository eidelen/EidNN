#include "widget.h"
#include "ui_widget.h"
#include "layer.h"

#include "QPixmap"
#include "QImage"


Widget::Widget(QWidget* parent) : QMainWindow(parent), ui(new Ui::Widget)
{
    ui->setupUi(this);

    // load mnist data set
    m_mnist = mnist::read_dataset<std::vector, std::vector, double, uint8_t>();
    m_currentIdx = 0;
    displayMNISTImage( m_currentIdx );

    connect( ui->formerSample, &QPushButton::pressed, [=]( )
    {
        if( m_currentIdx == 0 )
            m_currentIdx = m_mnist.training_images.size() - 1;
        else
            m_currentIdx--;

        displayMNISTImage( m_currentIdx );
    });

    connect( ui->nextSample, &QPushButton::pressed, [=]( )
    {
        if( m_currentIdx == m_mnist.training_images.size() - 1 )
            m_currentIdx = 0;
        else
            m_currentIdx++;

        displayMNISTImage( m_currentIdx );
    });

    //learn();
    sameImage();

}

Widget::~Widget()
{
    // Note: Since smartpointers are used, objects get deleted automatically.
    delete ui;
}

void Widget::displayMNISTImage( const size_t& idx )
{
    Eigen::MatrixXd imgIn; uint8_t lableIn;
    loadMNISTSample( m_mnist.training_images, m_mnist.training_labels, idx, imgIn, lableIn );

    int img_size = 28;
    QImage img(img_size, img_size, QImage::Format_RGB32);
    for( int h = 0; h < img_size; h++ )
    {
        for( int w = 0; w < img_size; w++ )
        {
            uint8_t pixValue = uint8_t( round( imgIn(28*h+w,0) ) );
            img.setPixel(w, h, qRgb(pixValue, pixValue, pixValue));
        }
    }

    ui->imgLable->setPixmap( QPixmap::fromImage(img.scaled(140,140)) );
    ui->imgLable->show();

    ui->trainingLable->setText( QString::number(lableIn, 10) );
}

void Widget::sameImage()
{
    std::vector<unsigned int> map = {784,30,10};
    Network* net = new Network(map);

    Eigen::MatrixXd xIn; uint8_t lable;
    loadMNISTSample( m_mnist.training_images, m_mnist.training_labels, 0, xIn, lable );
    Eigen::MatrixXd yOut = Eigen::MatrixXd::Constant(10,1, 0.0);
    yOut( lable, 0 ) = 1.0;

    for( int k = 0; k < 10000; k++ )
    {
        net->gradientDescent(xIn,yOut,3.0);
    }

    Eigen::MatrixXd err = net->getOutputLayer()->getBackpropagationError();
    std::cout << "Error = " << err.norm() << ": " << err.transpose() << std::endl;

    net->feedForward(xIn);
    Eigen::MatrixXd outSignal = net->getOutputActivation();

    std::cout << "Outsignal = " << outSignal.transpose() << std::endl;
    std::cout << "Lable = " << int(lable)  << " : " << yOut.transpose() << std::endl;

    delete net;
}

void Widget::learn()
{

    std::vector<unsigned int> map = {784,30,10};
    Network* net = new Network(map);
    unsigned int nbrEpochs = 60;

    size_t trainingIdx = 0;

    for( unsigned int epoch = 0; epoch < nbrEpochs; epoch++ )
    {
        // training
        for( size_t k = 0; k < 10000; k++ )
        {
            Eigen::MatrixXd xIn; uint8_t lable;
            loadMNISTSample( m_mnist.training_images, m_mnist.training_labels, trainingIdx, xIn, lable );
            Eigen::MatrixXd yOut = Eigen::MatrixXd::Constant(10,1, 0.0);
            yOut( lable, 0 ) = 1.0;

            net->gradientDescent(xIn,yOut,3.0);

            trainingIdx++;
        }

        if( trainingIdx >= m_mnist.training_images.size() )
            trainingIdx = 0;


        // test
        double successfull = 0;
        for( size_t t = 0; t < m_mnist.test_images.size(); t++ )
        {
            Eigen::MatrixXd xIn;  uint8_t lable;
            loadMNISTSample( m_mnist.test_images, m_mnist.test_labels, t, xIn, lable );

            net->feedForward(xIn);
            Eigen::MatrixXd out = net->getOutputActivation();
            double maxElement = -1000.0;
            int maxIdx = 0;
            for( int j = 0; j < out.rows(); j++ )
            {
                if( maxElement < out(j,0) )
                {
                    maxElement =  out(j,0);
                    maxIdx = j;
                }
            }

            if( maxIdx == lable )// && maxElement > 0.7 )
                successfull = successfull + 1;


        }

        std::cout << "Epoch " << epoch <<  ": success rate = " << 100 * successfull / double(m_mnist.test_images.size()) << "%" << std::endl;
    }

    delete net;
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
