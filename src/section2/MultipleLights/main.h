#pragma once
#include "WindowApp.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

class MultipleLights : public Renderer::WindowApp
{
public:
    MultipleLights(int width = 1920, int height = 1080, const std::string &title = "MultipleLights", const std::string &cameraType = "tps")
        : WindowApp(width, height, title, cameraType)
    {
    }

    ~MultipleLights() = default;

    virtual void Init() override
    {
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
        spotLight = m_lightManager->CreateSpotLight("spotLight");
        directionalLight = m_lightManager->CreateDirectionalLight("directionalLight", lightDirection, glm::vec3(0.2f), glm::vec3(0.5f), glm::vec3(1.0f));
        for(int i = 0; i< 4; i++)
        {
            pointLight[i] = m_lightManager->CreatePointLight("pointLight" + std::to_string(i), pointLightPositions[i], glm::vec3(0.2f), glm::vec3(1.0f), glm::vec3(1.0f), 1.0f, 0.09f, 0.032f);
        }
        cubeShader = m_shaderManager->loadShader("cubeShader", SHADER_PATH "/MultipleLights/MultipleLights.vs", SHADER_PATH "/MultipleLights/MultipleLights.fs");
        lightShader = m_shaderManager->loadShader("lightShader", SHADER_PATH "/Colors/Colors.vs", SHADER_PATH "/Colors/Light.fs");
        m_shaderManager->bindShader(cubeShader);
        m_shaderManager->getShader(cubeShader)->setInt("material.diffuse", 0);
        m_shaderManager->getShader(cubeShader)->setInt("material.specular", 1);
        m_shaderManager->unbindShader();

        // you own init code here
        // set up vertex data (and buffer(s)) and configure vertex attributes
        // ------------------------------------------------------------------
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
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        // texture coord attribute
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)(6 * sizeof(float)));
        glEnableVertexAttribArray(2);

        glBindVertexArray(lightVAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        // note that we update the lamp's position attribute's stride to reflect the updated buffer data
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(0);
        // load ambient/diffuse texture
        diffuseMap = m_textureManager->loadTexture2D("ambientMap", ASSETS_PATH "/texture/109447235_p0.jpg");
        // load specular texture
        specularMap = m_textureManager->loadTexture2D("specularMap", ASSETS_PATH "/texture/93447255_p0.png");
        m_renderSystem->setCullFace(false, GL_BACK);
        spotLightPtr = reinterpret_cast<Renderer::SpotLight *>(m_lightManager->getLight(spotLight).get());
    }

    virtual void Render()
    {
        m_renderDevice->beginFrame();
        m_renderSystem->setClearColor(glm::vec4(m_BackColor, 1.0f));
        m_renderSystem->render();

        // render cube
        // get camera view matrix and projection matrix
        glm::mat4 view = m_camera->getViewMatrix();
        glm::mat4 projection = m_camera->getProjectionMatrix();
        // model = glm::rotate(model, glm::radians(-55.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        m_shaderManager->bindShader(cubeShader);
        Renderer::DirectionalLight *directionalLightPtr = reinterpret_cast<Renderer::DirectionalLight *>(m_lightManager->getLight(directionalLight).get());
        directionalLightPtr->setLightDirection(lightDirection);
        spotLightPtr->setLightDirection(m_camera->getFront());
        spotLightPtr->setLightPosition(m_camera->getPosition());
        // directional light
        m_lightManager->setLightUniform(directionalLight, m_shaderManager->getShader(cubeShader), m_camera, "dirLight");
        // point light
        for(int i = 0; i< 4; i++)
        {
            m_lightManager->setLightUniform(pointLight[i], m_shaderManager->getShader(cubeShader), m_camera, "pointLights", true, i);
        }
        m_lightManager->setLightUniform(spotLight, m_shaderManager->getShader(cubeShader), m_camera, "spotLight");
        m_shaderManager->getShader(cubeShader)->setMat4("view", view);
        m_shaderManager->getShader(cubeShader)->setMat4("projection", projection);
        // material
        m_textureManager->bindTexture(diffuseMap, 0);
        m_textureManager->bindTexture(specularMap, 1);
        m_shaderManager->getShader(cubeShader)->setFloat("material.shininess", materialShininess);

        glBindVertexArray(VAO);
        for (int i = 0; i < 10; i++)
        {
            glm::mat4 model = glm::mat4(1.0);
            model = glm::translate(model, cubePositions[i]);
            float angle = 20.0f * i;
            model = glm::rotate(model, glm::radians(angle), glm::normalize(glm::vec3(1.0f, 0.3f, 0.5f)));
            m_shaderManager->getShader(cubeShader)->setMat4("model", model);
            glDrawArrays(GL_TRIANGLES, 0, 36);
        }
        m_shaderManager->bindShader(lightShader);
        m_shaderManager->getShader(lightShader)->setMat4("view", view);
        m_shaderManager->getShader(lightShader)->setMat4("projection", projection);
        glm::mat4 model = glm::mat4(1.0);
        model = glm::translate(model, pointLightPosition);
        model = glm::scale(model, glm::vec3(0.2f));
        m_shaderManager->getShader(lightShader)->setVec3("lightColor", glm::vec3(1.0f));
        glBindVertexArray(VAO);
        for(int i = 0; i< 4; i++)
        {
            model = glm::mat4(1.0);
            model = glm::translate(model, pointLightPositions[i]);
            model = glm::scale(model, glm::vec3(0.2f));
            m_shaderManager->getShader(lightShader)->setMat4("model", model);
            glDrawArrays(GL_TRIANGLES, 0, 36);
        }
        m_shaderManager->unbindShader();
        DrawImGui();

        m_renderDevice->endFrame();
    }

    void DrawImGui()
    {
        // imgui
        {
            ImGui::Begin("LightCasters Example");
            ImGui::Text("LightCasters Example");
            ImGui::ColorEdit3("backgroundColor", (float *)&m_BackColor);
            ImGui::DragFloat("materialShininess", &materialShininess, 1.0f);
            ImGui::End();
        }
        {
            ImGui::Begin("light and material");
            ImGui::Text("light");
            ImGui::DragFloat3("lightDirection", (float *)&lightDirection, 0.1f);
            ImGui::Text("Point Light:");
            ImGui::DragFloat3("pointLightPosition", (float *)&pointLightPosition, 0.1f);
            ImGui::Text("Spot Light:");
            ImGui::DragFloat("spotLightLinear", spotLightPtr->getLinearPtr(), 0.001f, 0.0f, 1.0f);
            ImGui::DragFloat("spotLightQuadratic", spotLightPtr->getQuadraticPtr(), 0.0001f, 0.0f, 1.0f);
            ImGui::DragFloat("spotLightInnerCutoff", spotLightPtr->getInnerCutoffPtr(), 0.1f, glm::radians(0.0f), glm::radians(90.0f));
            ImGui::End();
        }
    }

    void Release() override
    {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteVertexArrays(1, &lightVAO);
    }

private:
    GLuint VAO, VBO, lightVAO;                      // vertex array object, vertex buffer object
    GLuint diffuseMap, specularMap;                 // texture
    GLuint cubeShader, lightShader;                 // shader
    GLuint pointLight[4], spotLight, directionalLight; // light
    float materialShininess{64.0f};
    Renderer::SpotLight *spotLightPtr;
    glm::vec3 pointLightPosition{glm::vec3(1.2f, 1.0f, 2.0f)};
    glm::vec3 lightDirection{glm::vec3(-0.2f, -1.0f, -0.3f)};
    glm::vec3 cubePositions[10]{
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
    glm::vec3 pointLightPositions[4]{
        glm::vec3(0.7f, 0.2f, 2.0f),
        glm::vec3(2.3f, -3.3f, -4.0f),
        glm::vec3(-4.0f, 2.0f, -12.0f),
        glm::vec3(0.0f, 0.0f, -3.0f)};
};
