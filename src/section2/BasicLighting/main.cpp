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
    PRINT_INFO("Start BasiLighting Example");
    auto window = Renderer::RenderDevice::getInstance();
    window->initialize("BasiLighting", 1920, 1080);
    // render system
    Renderer::RenderSystem::ptr renderSystem = window->getRenderSystem();
    // the render state is managed by render system
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
        // vertex           // normal
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
         0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,

        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
         0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,

        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
        -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
        -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,

         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
         0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
         0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,

        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
         0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,

        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f};
    GLuint shader1 = ShaderMgr->loadShader("cube", SHADER_PATH "/PhongLightingModel/PhongLighting.vs", SHADER_PATH "/PhongLightingModel/PhongLighting.fs");

    unsigned int VBO, VAO, lightVAO;
    glGenVertexArrays(1, &VAO);
    glGenVertexArrays(1, &lightVAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    // normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                          (void *)(3 * sizeof(float)));

    glBindVertexArray(lightVAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                          (void *)0);
    glEnableVertexAttribArray(0);
    GLuint shader2 = ShaderMgr->loadShader("light", SHADER_PATH "/Colors/Colors.vs", SHADER_PATH "/Colors/Light.fs");

    glm::vec3 toyColor(1.0f, 0.5f, 0.31f);
    glm::vec3 lightColor(1.0f, 1.0f, 1.0f);
    glm::vec3 backgroundColor(0.2f, 0.3f, 0.3f);
    
    glm::vec3 lightPos(1.2f, 1.0f, 2.0f);

    float ambientStrength = 0.2;
    float diffuseStrength = 0.5;
    float specularStength = 1.0;
    // we can use some set functions in renderSystem class to control render state. such like:
    renderSystem->setCullFace(false, GL_BACK); // here we disable cull face or we won't see the triangle in opengl render window
    while (window->run())
    {
        window->beginFrame();

        renderSystem->setClearColor(glm::vec4(backgroundColor, 1.0f));
        renderSystem->render();

        // get camera view matrix and projection matrix
        glm::mat4 view = camera->getViewMatrix();
        glm::mat4 projection = camera->getProjectionMatrix();
        glm::mat4 model(1.0);
        // model = glm::rotate(model, glm::radians(-55.0f), glm::vec3(1.0f, 0.0f, 0.0f));

        ShaderMgr->bindShader(shader1);
        ShaderMgr->getShader(shader1)->setMat4("model", model);
        ShaderMgr->getShader(shader1)->setMat4("view", view);
        ShaderMgr->getShader(shader1)->setMat4("projection", projection);
        // phong lighting model
        ShaderMgr->getShader(shader1)->setVec3("toyColor", toyColor);
        ShaderMgr->getShader(shader1)->setVec3("lightColor", lightColor);
        ShaderMgr->getShader(shader1)->setVec3("lightPos", lightPos);
        ShaderMgr->getShader(shader1)->setVec3("viewPos", camera->getPosition());

        ShaderMgr->getShader(shader1)->setFloat("ambientStrength", ambientStrength);
        ShaderMgr->getShader(shader1)->setFloat("diffuseStrength", diffuseStrength);
        ShaderMgr->getShader(shader1)->setFloat("specularStength", specularStength);

        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        // draw light
        ShaderMgr->bindShader(shader2);
        model = glm::translate(glm::mat4(1.0), lightPos);
        model = glm::scale(model, glm::vec3(0.2f));
        ShaderMgr->getShader(shader2)->setMat4("model", model);
        ShaderMgr->getShader(shader2)->setMat4("view", view);
        ShaderMgr->getShader(shader2)->setMat4("projection", projection);
        ShaderMgr->getShader(shader2)->setVec3("lightColor", lightColor);
        glBindVertexArray(lightVAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        ShaderMgr->unbindShader();

        // imgui
        {
            ImGui::Begin("BasiLighting Example");
            ImGui::Text("BasiLighting Example");
            ImGui::ColorEdit3("backgroundColor", (float *)&backgroundColor);
            ImGui::ColorEdit3("toyColor", (float *)&toyColor);
            ImGui::ColorEdit3("lightColor", (float *)&lightColor);
            ImGui::DragFloat("ambientStrength", &ambientStrength, 0.01f, 0.0f, 1.0f);
            ImGui::DragFloat("diffuseStrength", &diffuseStrength, 0.01f, 0.0f, 1.0f);
            ImGui::DragFloat("specularStength", &specularStength, 0.01f, 0.0f, 1.0f);
            ImGui::DragFloat3("lightPos", (float *)&lightPos, 0.01f);
            ImGui::Text("Camera Type: %s", camera_type.c_str());
            ImGui::End();
        }
        window->endFrame();
    }
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    window->shutdown();
    return 0;
}