#include <QtCore/QDebug>
#include <QtCore/QFile>
#include <cassert>
#include "myqdir.h"

MyQDir::MyQDir (const QString & path) :
    QDir (path)
{

}

bool MyQDir::empty ()
{
    QFileInfoList dirs = this->entryInfoList (QDir::AllDirs | QDir::NoDot | QDir::NoDotDot);
    int errorcount = 0;
    foreach (QFileInfo d, dirs) {
        MyQDir subdir (d.absoluteFilePath ());
        qDebug () << "Empty directory:" << d.absoluteFilePath ();
        errorcount += subdir.empty () ? 0 : 1;
        qDebug () << "Remove directory:" << d.absoluteFilePath ();
        errorcount += this->rmdir(d.baseName()) ? 0 : 1;
        //errorcount += subdir.rmdir (".") ? 0 : 1;
    }
    assert (0 == errorcount);

    QFileInfoList files = this->entryInfoList (QDir::Files | QDir::NoDot | QDir::NoDotDot);

    foreach (QFileInfo f, files) {
        qDebug () << "Remove file:" << f.absoluteFilePath ();
        errorcount += QFile::remove (f.absoluteFilePath ()) ? 0 : 1;
    }
    return (0 == errorcount ? true : false);
}
