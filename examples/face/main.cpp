/****************************************************************************
**
** Copyright (C) 2014 Advanced Oesteotomy Tools AG
** Contact: http://www.aot-swiss.com/
**
** Adrian Schneider, adrian.schneider@unibas.ch
**
****************************************************************************/

#include <QApplication>
#include "widget.h"

int main(int argc, char* argv[])
{
    QApplication a(argc, argv);
    Widget w;
    w.show();
    return a.exec();
}
