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
    PRINT_INFO("Start LightingMaps Example");
    auto window = Renderer::RenderDevice::getInstance();
    window->initialize("LightingMaps", 1920, 1080);
    // render system
    Renderer::RenderSystem::ptr renderSystem = window->getRenderSystem();
    // the render state is managed by render system
    Renderer::ShaderManager::ptr ShaderMgr = renderSystem->getShaderManager();
    Renderer::TextureManager::ptr TextureMgr = renderSystem->getTextureManager();
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
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f,
         0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f,  0.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f,  1.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f,  1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f,  1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f,

        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,
         0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f,  0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f,  1.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f,  1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f,  1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,

        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f,  0.0f,
        -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  1.0f,  1.0f,
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
        -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  0.0f,  0.0f,
        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f,  0.0f,

         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,
         0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f,  1.0f,
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
         0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f,  0.0f,
         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,

        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f,  1.0f,
         0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  1.0f,  1.0f,
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f,  0.0f,
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f,  0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  0.0f,  0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f,  1.0f,

        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  1.0f,  1.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f,  0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f,  0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  0.0f,  0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f
    };
    GLuint shader1 = ShaderMgr->loadShader("cube", SHADER_PATH "/LightingMaps/LightingMaps.vs", SHADER_PATH "/LightingMaps/LightingMaps.fs");
    ShaderMgr->bindShader(shader1);
    ShaderMgr->getShader(shader1)->setInt("material.diffuse",  0);
    ShaderMgr->getShader(shader1)->setInt("material.specular", 1);
    ShaderMgr->getShader(shader1)->setInt("material.emission", 2);
    ShaderMgr->unbindShader();

    unsigned int VBO, VAO, lightVAO;
    glGenVertexArrays(1, &VAO);
    glGenVertexArrays(1, &lightVAO);
    glGenBuffers(1, &VBO);

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
    //texture attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                         (void *)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(lightVAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                          (void *)0);
    glEnableVertexAttribArray(0);
    GLuint shader2 = ShaderMgr->loadShader("light", SHADER_PATH "/Colors/Colors.vs", SHADER_PATH "/Colors/Light.fs");

    // load ambient/diffuse texture
    GLuint ambientMap = TextureMgr->loadTexture2D("ambientMap", ASSETS_PATH "/texture/109447235_p0.jpg");
    // load specular texture
    GLuint specularMap = TextureMgr->loadTexture2D("specularMap", ASSETS_PATH "/texture/93447255_p0.png");
    // load emission texture
    GLuint emissionMap = TextureMgr->loadTexture2D("emissionMap", ASSETS_PATH "/texture/matrix.jpg");

    glm::vec3 lightColor(1.0f, 1.0f, 1.0f);
    glm::vec3 backgroundColor(0.2f, 0.3f, 0.3f);
    
    glm::vec3 lightPos(0.2f, 1.0f, 0.5f);

    glm::vec3 materialSpecular(0.5f, 0.5f, 0.5f);
    float materialShininess = 64.0f;
    float emissionStrength = 0.5f;

    glm::vec3 lightAmbient(0.2f, 0.2f, 0.2f);
    glm::vec3 lightDiffuse(0.5f, 0.5f, 0.5f);
    glm::vec3 lightSpecular(1.0f, 1.0f, 1.0f);

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
        model = glm::rotate(model, glm::radians(-55.0f), glm::vec3(1.0f, 0.0f, 0.0f));

        ShaderMgr->bindShader(shader1);
        ShaderMgr->getShader(shader1)->setMat4("model", model);
        ShaderMgr->getShader(shader1)->setMat4("view", view);
        ShaderMgr->getShader(shader1)->setMat4("projection", projection);
        // phong lighting model
        ShaderMgr->getShader(shader1)->setVec3("lightColor", lightColor);
        ShaderMgr->getShader(shader1)->setVec3("light.position", lightPos);
        ShaderMgr->getShader(shader1)->setVec3("viewPos", camera->getPosition());

        // material
        TextureMgr->bindTexture(ambientMap, 0);
        TextureMgr->bindTexture(specularMap, 1);
        TextureMgr->bindTexture(emissionMap, 2);
        ShaderMgr->getShader(shader1)->setVec3("material.specular", materialSpecular);
        ShaderMgr->getShader(shader1)->setFloat("material.shininess", materialShininess);
        ShaderMgr->getShader(shader1)->setFloat("emissionStrength", emissionStrength);
        ShaderMgr->getShader(shader1)->setFloat("emissionTime", (double)glfwGetTime());
        // light
        ShaderMgr->getShader(shader1)->setVec3("light.ambient", lightAmbient);
        ShaderMgr->getShader(shader1)->setVec3("light.diffuse", lightDiffuse);
        ShaderMgr->getShader(shader1)->setVec3("light.specular", lightDiffuse);
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
            ImGui::Begin("LightingMaps Example");
            ImGui::Text("LightingMaps Example");
            ImGui::ColorEdit3("backgroundColor", (float *)&backgroundColor);
            ImGui::ColorEdit3("lightColor", (float *)&lightColor);
            ImGui::DragFloat3("lightPos", (float *)&lightPos, 0.01f);
            ImGui::DragFloat("materialShininess", &materialShininess, 0.1f);
            ImGui::DragFloat("emissionStrength", &emissionStrength, 0.1f);
            ImGui::Text("Camera Type: %s", camera_type.c_str());
            ImGui::End();
        }

        {
            ImGui::Begin("light and material");
            ImGui::Text("light");
            ImGui::ColorEdit3("light.ambient", (float *)&lightAmbient);
            ImGui::ColorEdit3("light.diffuse", (float *)&lightDiffuse);
            ImGui::ColorEdit3("light.specular", (float *)&lightSpecular);
            ImGui::End();
        }
        window->endFrame();
    }
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    window->shutdown();
    return 0;
}