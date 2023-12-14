#include "RenderApp/RenderDevice.h"
#include "ColorfulPrint.h"
#include "Config.h"
#include "Manager/ShaderManager.h"

int main()
{
    PRINT_INFO("Hello, colorful triangle!");
    std::shared_ptr<Renderer::RenderDevice> window = Renderer::RenderDevice::getInstance();
    window->initialize("Colorful Triangle", 1920, 1080);
    glm::vec3 backgroundColor(0.2f, 0.3f, 0.3f);

    float vertices[] = {
        -0.5f, -0.5f, 0.0f, 1.0, 0.0, 0.0, // left 
         0.5f, -0.5f, 0.0f, 0.0, 1.0, 0.0,// right
         0.0f,  0.5f, 0.0f, 0.0, 0.0, 1.0// top
    };
    Renderer::ShaderManager::ptr shaderManager = Renderer::ShaderManager::getInstance();
    std::string vertexShaderPath(SHADER_PATH);
    vertexShaderPath += "/colorfulTriangle/colorfulTriangle.vert";
    std::string fragmentShaderPath(SHADER_PATH);
    fragmentShaderPath += "/colorfulTriangle/colorfulTriangle.frag";
    unsigned int TriangleShader = shaderManager->loadShader("TriangleShader", vertexShaderPath, fragmentShaderPath);

    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO);
    // 1. bind Vertex Array Object
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // 2. copy our vertices array in a buffer for OpenGL to use
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    // 3. then set our vertex attributes pointers
    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    while (window->run())
    {
        window->beginFrame();

        glClearColor(backgroundColor.x, backgroundColor.y, backgroundColor.z, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        shaderManager->bindShader(TriangleShader);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindVertexArray(0);
        {
            ImGui::Begin("Colorful Triangle");
            ImGui::ColorEdit3("Background Color", (float *)&backgroundColor);
            ImGui::End();
        }
        window->endFrame();
    }
    window->shutdown();
    return 0;
}