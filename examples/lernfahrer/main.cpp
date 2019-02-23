/****************************************************************************
**
** Copyright (C) 2014 Advanced Oesteotomy Tools AG
** Contact: http://www.aot-swiss.com/
**
** Adrian Schneider, adrian.schneider@unibas.ch
**
****************************************************************************/

#include "window.h"
#include <QApplication>
#include <QSurfaceFormat>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    QSurfaceFormat fmt;
    fmt.setSamples(4);
    QSurfaceFormat::setDefaultFormat(fmt);

    Window window;
    window.show();
    return app.exec();
}