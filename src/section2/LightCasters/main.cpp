#include "RenderApp/RenderDevice.h"
#include "Manager/TextureManager.h"
#include "Manager/ShaderManager.h"
#include "ColorfulPrint.h"
#include "Config.h"
#include "Camera/TPSCamera.h"
#include "Camera/FPSCamera.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

int main(int argc, char **argv)
{
    PRINT_INFO("Start LightCasters Example");
    auto window = Renderer::RenderDevice::getInstance();
    window->initialize("LightCasters", 1920, 1080);
    // render system
    Renderer::RenderSystem::ptr renderSystem = window->getRenderSystem();
    // the render state is managed by render system
    Renderer::ShaderManager::ptr ShaderMgr = renderSystem->getShaderManager();
    Renderer::TextureManager::ptr TextureMgr = renderSystem->getTextureManager();
    Renderer::LightManager::ptr LightMgr = renderSystem->getLightManager();
    unsigned int dirlight = LightMgr->CreateDirectionalLight("dirLight", glm::vec3(-0.2f, -1.0f, -0.3f), glm::vec3(0.2f), glm::vec3(0.5f), glm::vec3(1.0f));
    unsigned int pointlight = LightMgr->CreatePointLight("pointLight", glm::vec3(0.2f, 2.7f, 0.2f), glm::vec3(0.2f), glm::vec3(0.5f), glm::vec3(1.0f), 1.0f, 0.09f, 0.032f);
    unsigned int spotlight = LightMgr->CreateSpotLight("spotLight");
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
        // positions          // normals           // texture coords
        -0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f,
        0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f,
        0.5f, 0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f,
        0.5f, 0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f,
        -0.5f, 0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f,

        -0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f,
        0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
        0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
        -0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,
        -0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,

        -0.5f, 0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
        -0.5f, 0.5f, -0.5f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
        -0.5f, -0.5f, -0.5f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
        -0.5f, -0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        -0.5f, 0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f,

        0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
        0.5f, 0.5f, -0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
        0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
        0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
        0.5f, -0.5f, 0.5f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,

        -0.5f, -0.5f, -0.5f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f,
        0.5f, -0.5f, -0.5f, 0.0f, -1.0f, 0.0f, 1.0f, 1.0f,
        0.5f, -0.5f, 0.5f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f,
        0.5f, -0.5f, 0.5f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f,
        -0.5f, -0.5f, 0.5f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f,

        -0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
        0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f,
        0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
        0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
        -0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        -0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
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
    GLuint shader1 = ShaderMgr->loadShader("cube", SHADER_PATH "/LightCasters/LightCasters.vs", SHADER_PATH "/LightCasters/LightCasters.fs");
    GLuint lightCube = ShaderMgr->loadShader("lightCube", SHADER_PATH "/Colors/Colors.vs", SHADER_PATH "/Colors/Light.fs");
    ShaderMgr->bindShader(shader1);
    ShaderMgr->getShader(shader1)->setInt("material.diffuse", 0);
    ShaderMgr->getShader(shader1)->setInt("material.specular", 1);
    ShaderMgr->unbindShader();

    unsigned int VBO, VAO, lightVAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenVertexArrays(1, &lightVAO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    // normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                          (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                          (void *)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(lightVAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    // load ambient/diffuse texture
    GLuint ambientMap = TextureMgr->loadTexture2D("ambientMap", ASSETS_PATH "/texture/109447235_p0.jpg");
    // load specular texture
    GLuint specularMap = TextureMgr->loadTexture2D("specularMap", ASSETS_PATH "/texture/93447255_p0.png");

    glm::vec3 backgroundColor(0.0, 0.0, 0.0);
    glm::vec3 lightDirection(-0.2f, -1.0f, -0.3f);
    glm::vec3 pointLightPosition(0.2f, 2.7f, 0.2f);

    float materialShininess = 64.0f;
    // we can use some set functions in renderSystem class to control render state. such like:
    renderSystem->setCullFace(false, GL_BACK); // here we disable cull face or we won't see the triangle in opengl render window
    auto spotLight = reinterpret_cast<Renderer::SpotLight *>(LightMgr->getLight(spotlight).get());
    while (window->run())
    {
        window->beginFrame();

        renderSystem->setClearColor(glm::vec4(backgroundColor, 1.0f));
        renderSystem->render();

        // get camera view matrix and projection matrix
        glm::mat4 view = camera->getViewMatrix();
        glm::mat4 projection = camera->getProjectionMatrix();
        // model = glm::rotate(model, glm::radians(-55.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        ShaderMgr->bindShader(shader1);
        // light
        spotLight->setLightDirection(camera->getFront());
        spotLight->setLightPosition(camera->getPosition());
        spotLight->setLightUniforms(ShaderMgr->getShader(shader1), camera, "spotLight");
        // LightMgr->setLightUniform(spotlight, ShaderMgr->getShader(shader1), camera);
        // camera
        ShaderMgr->getShader(shader1)->setMat4("view", view);
        ShaderMgr->getShader(shader1)->setMat4("projection", projection);
        // material
        TextureMgr->bindTexture(ambientMap, 0);
        TextureMgr->bindTexture(specularMap, 1);
        ShaderMgr->getShader(shader1)->setFloat("material.shininess", materialShininess);

        glBindVertexArray(VAO);
        for (int i = 0; i < 10; i++)
        {
            glm::mat4 model = glm::mat4(1.0);
            model = glm::translate(model, cubePositions[i]);
            float angle = 20.0f * i;
            model = glm::rotate(model, glm::radians(angle), glm::normalize(glm::vec3(1.0f, 0.3f, 0.5f)));
            ShaderMgr->getShader(shader1)->setMat4("model", model);
            glDrawArrays(GL_TRIANGLES, 0, 36);
        }

        ShaderMgr->bindShader(lightCube);
        ShaderMgr->getShader(lightCube)->setMat4("view", view);
        ShaderMgr->getShader(lightCube)->setMat4("projection", projection);
        glm::mat4 model = glm::mat4(1.0);
        model = glm::translate(model, pointLightPosition);
        model = glm::scale(model, glm::vec3(0.2f));
        ShaderMgr->getShader(lightCube)->setMat4("model", model);
        ShaderMgr->getShader(lightCube)->setVec3("lightColor", glm::vec3(1.0f));
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        ShaderMgr->unbindShader();
        // imgui
        {
            ImGui::Begin("LightCasters Example");
            ImGui::Text("LightCasters Example");
            ImGui::ColorEdit3("backgroundColor", (float *)&backgroundColor);
            ImGui::DragFloat("materialShininess", &materialShininess, 1.0f);
            ImGui::Text("Camera Type: %s", camera_type.c_str());
            ImGui::End();
        }

        {
            ImGui::Begin("light and material");
            ImGui::Text("light");
            ImGui::DragFloat3("lightDirection", (float *)&lightDirection, 0.1f);
            ImGui::Text("Point Light:");
            ImGui::DragFloat3("pointLightPosition", (float *)&pointLightPosition, 0.1f);
            ImGui::Text("Spot Light:");
            ImGui::DragFloat("spotLightLinear", spotLight->getLinearPtr(), 0.001f, 0.0f, 1.0f);
            ImGui::DragFloat("spotLightQuadratic", spotLight->getQuadraticPtr(), 0.0001f, 0.0f, 1.0f);
            ImGui::DragFloat("spotLightInnerCutoff", spotLight->getInnerCutoffPtr(), 0.1f, glm::radians(0.0f), glm::radians(90.0f));
            ImGui::End();
        }
        window->endFrame();
    }
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    window->shutdown();
    return 0;
}