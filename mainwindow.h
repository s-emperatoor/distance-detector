#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "cvjobs.h"
#include <QMutex>
#include <QTimer>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

public slots:
    void _onTimeout();

private:
    Ui::MainWindow *ui;
    CVJobs *job;
    QTimer *timer;


    //func

};
#endif // MAINWINDOW_H
