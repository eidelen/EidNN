#include "widget.h"
#include "ui_widget.h"
#include "mnist/mnist_reader.hpp"

#include "QPixmap"
#include "QImage"
#include <vector>

Widget::Widget(QWidget* parent) : QMainWindow(parent), ui(new Ui::Widget)
{
    ui->setupUi(this);

    showAImage();
}

Widget::~Widget()
{
    // Note: Since smartpointers are used, objects get deleted automatically.
    delete ui;
}

void Widget::showAImage()
{
    mnist::MNIST_dataset< std::vector, std::vector<uint8_t>, uint8_t> ds = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    std::vector<uint8_t> oneImage = ds.training_images.at(39);

    size_t i_size = oneImage.size();


    int img_size = 28;
    QImage img(img_size, img_size, QImage::Format_RGB32);
    for( int h = 0; h < img_size; h++ )
    {
        for( int w = 0; w < img_size; w++ )
        {
            uint8_t pixValue = oneImage.at(28*h+w);
            img.setPixel(w, h, qRgb(pixValue, pixValue, pixValue));
        }
    }

    ui->imgLable->setPixmap( QPixmap::fromImage(img.scaled(100,100)) );
    ui->imgLable->show();
}
