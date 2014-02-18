#ifndef MYQDIR_H
#define MYQDIR_H

#include <QtCore/QDir>
#include <QtCore/QString>

class MyQDir : public QDir
{

public:
    MyQDir ( const QString & path = QString() );

    bool empty ();

private:
    
};

#endif // MYQDIR_H
