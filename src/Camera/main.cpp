/**
 * @file main.cpp
 * @author Yuanmu Xu (xuyuanmu@outlook.com)
 * @brief OpenGL camera example: this example is much like the coordinate system example, but we use camera to view the scene. And we implement a render system to manage the render state.
 * @version 0.1
 * @date 2023-11-26
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "RenderDevice.h"
#include "TextureManager.h"
#include "ShaderManager.h"
#include "ColorfulPrint.h"
#include "Config.h"
#include "TPSCamera.h"
#include "FPSCamera.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

int main(int argc, char **argv)
{
    PRINT_INFO("Start CoordinateSystems Example");
    auto window = Renderer::RenderDevice::getInstance();
    window->initialize("Transformations", 1920, 1080);
    // render system
    Renderer::RenderSystem::ptr renderSystem = window->getRenderSystem();
    // the render state is managed by render system
    Renderer::TextureManager::ptr TexMgr = renderSystem->getTextureManager();
    Renderer::ShaderManager::ptr ShaderMgr = renderSystem->getShaderManager();
    // prcess args
    std::string camera_type = "tps";
    if (argc > 1)
        camera_type = argv[1];
    Renderer::Camera3D::ptr camera;
    if (camera_type == "tps")
    {
        PRINT_INFO("Use TPS Camera");
        // TPS camera
        camera = renderSystem->createTPSCamera(glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.0, 0.0, 0.0));
        camera->setPerspective(45.0f, static_cast<float>(window->getWindowWidth()) / window->getWindowHeight(), 0.1f, 100.0f);
        Renderer::TPSCamera *tpsCamera = reinterpret_cast<Renderer::TPSCamera *>(camera.get());
        tpsCamera->setPitch(15.0f);
        tpsCamera->setDistance(3.0f);
        tpsCamera->setDistanceLimit(0.01f, 1000.0f);
        tpsCamera->setWheelSensitivity(1.0f);
        tpsCamera->setMouseSensitivity(0.3f);
    }
    else if (camera_type == "fps")
    {
        PRINT_INFO("Use FPS Camera");
        // FPS camera
        camera = renderSystem->createFPSCamera(glm::vec3(0.0, 0.0, -3.0), glm::vec3(0.0, 0.0, 0.0));
        camera->setPerspective(45.0f, static_cast<float>(window->getWindowWidth()) / window->getWindowHeight(), 0.1f, 100.0f);
        Renderer::FPSCamera *fpsCamera = reinterpret_cast<Renderer::FPSCamera *>(camera.get());
        fpsCamera->setMouseSensitivity(0.3f);
        fpsCamera->setMoveSpeed(1.f);
    }
    else
    {
        PRINT_ERROR("Unknown camera type: " << camera_type);
        return -1;
    }

    float vertices[] = {
        -0.5f, -0.5f, -0.5f, 0.0f, 0.0f,
        0.5f, -0.5f, -0.5f, 1.0f, 0.0f,
        0.5f, 0.5f, -0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, -0.5f, 1.0f, 1.0f,
        -0.5f, 0.5f, -0.5f, 0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f, 0.0f, 0.0f,

        -0.5f, -0.5f, 0.5f, 0.0f, 0.0f,
        0.5f, -0.5f, 0.5f, 1.0f, 0.0f,
        0.5f, 0.5f, 0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, 0.5f, 1.0f, 1.0f,
        -0.5f, 0.5f, 0.5f, 0.0f, 1.0f,
        -0.5f, -0.5f, 0.5f, 0.0f, 0.0f,

        -0.5f, 0.5f, 0.5f, 1.0f, 0.0f,
        -0.5f, 0.5f, -0.5f, 1.0f, 1.0f,
        -0.5f, -0.5f, -0.5f, 0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f, 0.0f, 1.0f,
        -0.5f, -0.5f, 0.5f, 0.0f, 0.0f,
        -0.5f, 0.5f, 0.5f, 1.0f, 0.0f,

        0.5f, 0.5f, 0.5f, 1.0f, 0.0f,
        0.5f, 0.5f, -0.5f, 1.0f, 1.0f,
        0.5f, -0.5f, -0.5f, 0.0f, 1.0f,
        0.5f, -0.5f, -0.5f, 0.0f, 1.0f,
        0.5f, -0.5f, 0.5f, 0.0f, 0.0f,
        0.5f, 0.5f, 0.5f, 1.0f, 0.0f,

        -0.5f, -0.5f, -0.5f, 0.0f, 1.0f,
        0.5f, -0.5f, -0.5f, 1.0f, 1.0f,
        0.5f, -0.5f, 0.5f, 1.0f, 0.0f,
        0.5f, -0.5f, 0.5f, 1.0f, 0.0f,
        -0.5f, -0.5f, 0.5f, 0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f, 0.0f, 1.0f,

        -0.5f, 0.5f, -0.5f, 0.0f, 1.0f,
        0.5f, 0.5f, -0.5f, 1.0f, 1.0f,
        0.5f, 0.5f, 0.5f, 1.0f, 0.0f,
        0.5f, 0.5f, 0.5f, 1.0f, 0.0f,
        -0.5f, 0.5f, 0.5f, 0.0f, 0.0f,
        -0.5f, 0.5f, -0.5f, 0.0f, 1.0f};

    glm::vec3 cubePositions[] = {
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(2.0f, 5.0f, -15.0f),
        glm::vec3(-1.5f, -2.2f, -2.5f),
        glm::vec3(-3.8f, -2.0f, -12.3f),
        glm::vec3(2.4f, -0.4f, -3.5f),
        glm::vec3(-1.7f, 3.0f, -7.5f),
        glm::vec3(1.3f, -2.0f, -2.5f),
        glm::vec3(1.5f, 2.0f, -2.5f),
        glm::vec3(1.5f, 0.2f, -1.5f),
        glm::vec3(-1.3f, 1.0f, -1.5f)};

    GLuint shader1 = ShaderMgr->loadShader("texture", SHADER_PATH "/CoordinateSystems/CoordinateSystems.vs", SHADER_PATH "/CoordinateSystems/CoordinateSystems.fs");

    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
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
    // we can use some set functions in renderSystem class to control render state. such like:
    renderSystem->setCullFace(false, GL_BACK); // here we disable cull face or we won't see the triangle in opengl render window
    while (window->run())
    {
        window->beginFrame();

        renderSystem->setClearColor(glm::vec4(backgroundColor, 1.0f));
        renderSystem->render();
        // bind texture
        ShaderMgr->bindShader(shader1);
        glm::mat4 model(1.0);
        model = glm::rotate(model, glm::radians(-55.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        glm::mat4 view = camera->getViewMatrix();
        glm::mat4 projection = camera->getProjectionMatrix();
        // get camera view matrix and projection matrix
        TexMgr->bindTexture(tex1, 0);
        TexMgr->bindTexture(tex2, 1);
        ShaderMgr->getShader(shader1)->setFloat("mixValue", mixValue);
        ShaderMgr->getShader(shader1)->setMat4("view", view);
        ShaderMgr->getShader(shader1)->setMat4("projection", projection);
        glBindVertexArray(VAO);
        for (unsigned int i = 0; i < 10; i++)
        {
            glm::mat4 model(1.0f);
            model = glm::translate(model, cubePositions[i]);
            float angle = 20.0f * i;
            model = glm::rotate(model, glm::radians(angle), glm::vec3(1.0f, 0.3f, 0.5f));
            ShaderMgr->getShader(shader1)->setMat4("model", model);

            glDrawArrays(GL_TRIANGLES, 0, 36);
        }
        ShaderMgr->unbindShader();
        glBindVertexArray(0);

        // imgui
        {
            ImGui::Begin("Transformations");
            ImGui::Text("Transformation Example");
            ImGui::ColorEdit3("backgroundColor", (float *)&backgroundColor);
            ImGui::SliderFloat3("scale", (float *)&scale, 0.0f, 2.0f);
            ImGui::SliderFloat("mixValue", &mixValue, 0.0f, 1.0f);
            ImGui::End();
        }
        window->endFrame();
    }
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    window->shutdown();
    return 0;
}