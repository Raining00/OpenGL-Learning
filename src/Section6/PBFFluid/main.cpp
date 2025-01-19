#include "main.h"

int main(int argc, char **argv)
{
    std::string cameraType = "tps";
    if(argc > 1)
        cameraType = argv[1];
    PBF app(1920, 1080, "PBF", cameraType);
    app.Run();
    return 0;
}