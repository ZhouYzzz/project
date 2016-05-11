#include "iostream"
#include "cstdio"

using namespace std;

int main(int argc, char const *argv[])
{
    string a = "asd";
    char b[20];
    sprintf(b, "%08d.jpg", 1);
    cout << a + string(b) << endl;
    return 0;
}