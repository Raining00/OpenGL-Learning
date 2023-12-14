#include "RenderApp/RenderDevice.h"
#include "ColorfulPrint.h"

int main()
{
    PRINT_INFO("Test window class!");
    std::shared_ptr<Renderer::RenderDevice> window = Renderer::RenderDevice::getInstance();
    window->initialize("TestWindowClass", 1920, 1080, false);
    glm::vec3 backgroundColor(0.2f, 0.3f, 0.3f);
    while(window->run())
    {
        window->beginFrame();

        // set background color
        glClearColor(backgroundColor.x, backgroundColor.y, backgroundColor.z, 1.0f);
        // clear the buffer
        glClear(GL_COLOR_BUFFER_BIT);
        {
            ImGui::Begin("TestWindowClass");
            ImGui::Text("Hello World!");
            ImGui::ColorEdit3("clear color", (float*)&backgroundColor);
            ImGui::End();
        }
        window->endFrame();
    }
    window->shutdown();
    return 0;
}