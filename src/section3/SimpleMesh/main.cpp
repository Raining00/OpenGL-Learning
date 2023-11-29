#include "main.h"

int main(int argc, char **argv)
{
    MultipleLights app(1920, 1080, "SimpleMesh", "tps");
    app.Run();
    return 0;
}