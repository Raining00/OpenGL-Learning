#include "main.h"

int main(int argc, char **argv)
{
    std::string cameraType = "tps";
    if(argc > 1)
        cameraType = argv[1];
    ExplodeObject app(1920, 1080, "ExplodeObject", cameraType);
    app.Run();
    return 0;
}