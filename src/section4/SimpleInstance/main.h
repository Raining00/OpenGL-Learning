#pragma once
#include "WindowApp.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "StaticModelDrawable.h"

class SimpleInstance : public Renderer::WindowApp
{
public:
    GLuint vao, vbo, instanceVBO;

    SimpleInstance(int width = 1920, int height = 1080, const std::string& title = "SimpleInstance", const std::string& cameraType = "tps")
        : WindowApp(width, height, title, cameraType)
    {
    }

    ~SimpleInstance() = default;

    virtual void Init() override
    {
        float quadVertices[] =
        {
            -0.05f, -0.05f,  0.0f, 0.0f, 1.0f,
             0.05f, -0.05f,  0.0f, 1.0f, 0.0f,
            -0.05f,  0.05f,  1.0f, 0.0f, 0.0f,

            -0.05f,  0.05f,  1.0f, 0.0f, 0.0f,
             0.05f, -0.05f,  0.0f, 1.0f, 0.0f,
             0.05f,  0.05f,  0.0f, 1.0f, 1.0f
        };
        myShader = m_shaderManager->loadShader("SimpleInstance", SHADER_PATH"/SimpleInstance/SimpleInstance.vs", SHADER_PATH"/SimpleInstance/SimpleInstance.fs");

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        int index = 0;
        float offset = 0.1f;
        for (int y = -10; y < 10; y += 2)
        {
            for (int x = -10; x < 10; x += 2)
            {
                glm::vec2 translation;
                translation.x = (float)x / 10.0f + offset;
                translation.y = (float)y / 10.0f + offset;
                translations[index++] = translation;
            }
        }

        glGenBuffers(1, &instanceVBO);
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * 100, &translations[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glVertexAttribDivisor(2, 1);
        //m_shaderManager->bindShader(myShader);
        //for (unsigned int i = 0; i < 100; i++)
        //{
        //    std::stringstream ss;
        //    std::string index;
        //    ss << i;
        //    index = ss.str();
        //    m_shaderManager->getShader(myShader)->setVec2("offsets[" + index + "]", translations[i]);
        //}
    }

    virtual void Render() override
    {
        if (m_enablePolygonMode)
            m_renderSystem->setPolygonMode(GL_LINE);
        else
            m_renderSystem->setPolygonMode(GL_FILL);
        m_renderDevice->beginFrame();
        m_renderSystem->setClearColor(glm::vec4(m_BackColor, 1.0f));
        m_renderSystem->setSunLight(sunLightDir, glm::vec3(ambientCoef), glm::vec3(diffuseCoef), glm::vec3(specularCoef));
        m_renderSystem->render();
        m_shaderManager->bindShader(myShader);
        glBindVertexArray(vao);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, 100);
        m_shaderManager->unbindShader();
        DrawImGui();
        m_renderDevice->endFrame();
    }

    void DrawImGui()
    {
        // imgui
        {
            ImGui::Begin("SimpleInstance Example");
            ImGui::Text("SimpleInstance Example");
            ImGui::ColorEdit3("backgroundColor", (float*)&m_BackColor);
            ImGui::Text("SunLight");
            ImGui::SliderFloat3("sunLightDir", (float*)&sunLightDir, -1.0f, 1.0f);
            ImGui::ColorEdit3("sunLightColor", (float*)&sunLightColor);
            ImGui::Text("Ambient");
            ImGui::SliderFloat("ambientColor", (float*)&ambientCoef, 0.0f, 1.0f);
            ImGui::Text("Diffuse");
            ImGui::SliderFloat("diffuseColor", (float*)&diffuseCoef, 0.0, 1.0f);
            ImGui::Text("Specular");
            ImGui::SliderFloat("specularColor", (float*)&specularCoef, 0.0, 1.0f);
            ImGui::Checkbox("enableDepthTest", &m_enableDepthTest);
            ImGui::Checkbox("PolygonMode", &m_enablePolygonMode);
            ImGui::End();
        }
    }

    void Release() override
    {
    }

private:
    glm::vec2 translations[100];
    bool m_enableDepthTest, m_enablePolygonMode{false};
    GLuint myShader;
    glm::vec3 sunLightDir{ glm::vec3(0.072f, 0.42f, 1.0f) };
    glm::vec3 sunLightColor{ glm::vec3(0.5) };
    float ambientCoef{ 0.5f };
    float diffuseCoef{ 0.6f };
    float specularCoef{ 0.6f };
};

