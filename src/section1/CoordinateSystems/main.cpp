#include "RenderDevice.h"
#include "TextureManager.h"
#include "ShaderManager.h"
#include "ColorfulPrint.h"
#include "Config.h"
#include "TPSCamera.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

int main()
{
    PRINT_INFO("Start CoordinateSystems Example");
    auto window = Renderer::RenderDevice::getInstance();
    window->initialize("CoordinateSystems", 1920, 1080);
    Renderer::RenderSystem::ptr renderSystem = window->getRenderSystem();
    Renderer::TextureManager::ptr TexMgr = renderSystem->getTextureManager();
    Renderer::ShaderManager::ptr ShaderMgr = renderSystem->getShaderManager();

    Renderer::Camera3D::ptr camera = renderSystem->createTPSCamera(glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.0, 0.0, 0.0));
    camera->setPerspective(45.0f, static_cast<float>(window->getWindowWidth()) / window->getWindowHeight(), 0.1f, 100.0f);
    // Renderer::TPSCamera *tpsCamera = reinterpret_cast<Renderer::TPSCamera*>(camera.get());
    // tpsCamera->setPitch(15.0f);
    // tpsCamera->setDistance(3.0f);
    // tpsCamera->setDistanceLimit(0.01f, 1000.0f);
    // tpsCamera->setWheelSensitivity(5.0f);
    // tpsCamera->setMouseSensitivity(0.3f);

    float vertices[] = {
        //      ---- 位置 ----       ---- 颜色 ----     - 纹理坐标 -
        0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,   
        0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,  
        -0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 
        -0.5f, 0.5f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f   
    };

    unsigned int indices[] = {
        0, 1, 3, // 第一个三角形
        1, 2, 3  // 第二个三角形
    };

    GLuint shader1 = ShaderMgr->loadShader("texture", SHADER_PATH "/CoordinateSystems/CoordinateSystems.vs", SHADER_PATH "/CoordinateSystems/CoordinateSystems.fs");

    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // create texture
    GLuint tex1 = TexMgr->loadTexture2D("tomorin", ASSETS_PATH "/texture/109447235_p0.jpg");
    GLuint tex2 = TexMgr->loadTexture2D("pic2", ASSETS_PATH "/texture/93447255_p0.png");
    glm::vec3 backgroundColor(0.2f, 0.3f, 0.3f);
    glm::vec3 scale(1.0f, 1.0f, 1.0f);
    float mixValue = 0.2f;
    ShaderMgr->bindShader(shader1);
    ShaderMgr->getShader(shader1)->setInt("texture1", 0);
    ShaderMgr->getShader(shader1)->setInt("texture2", 1);
    ShaderMgr->unbindShader();
    while(window->run())
    {
        window->beginFrame();
        glClearColor(backgroundColor.r, backgroundColor.g, backgroundColor.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // bind texture
        ShaderMgr->bindShader(shader1);
        glm::mat4 model(1.0);
        model = glm::rotate(model, glm::radians(-55.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        glm::mat4 view(1.0);
        view = glm::translate(view, glm::vec3(0.0f, 0.0f, -3.0f));
        glm::mat4 projection(1.0);
        projection = glm::perspective(glm::radians(45.0f), (float)window->getWindowWidth() / (float)window->getWindowHeight(), 0.1f, 100.0f);
        ShaderMgr->getShader(shader1)->setMat4("model", model);
        ShaderMgr->getShader(shader1)->setMat4("view", view);
        ShaderMgr->getShader(shader1)->setMat4("projection", projection);        

        TexMgr->bindTexture(tex1, 0);
        TexMgr->bindTexture(tex2, 1);
        ShaderMgr->getShader(shader1)->setFloat("mixValue", mixValue);
        // draw triangle
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        ShaderMgr->unbindShader();
        
        // imgui
        {
            ImGui::Begin("CoordinateSystems Example");
            ImGui::Text("CoordinateSystems Example");
            ImGui::ColorEdit3("backgroundColor", (float*)&backgroundColor);
            ImGui::SliderFloat3("scale", (float*)&scale, 0.0f, 2.0f);
            ImGui::SliderFloat("mixValue", &mixValue, 0.0f, 1.0f);
            ImGui::End();
        }
        window->endFrame();
    }
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteVertexArrays(1, &VAO);
    window->shutdown();
    return 0;
}