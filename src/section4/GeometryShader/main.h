#pragma once
#include "WindowApp.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "StaticModelDrawable.h"

class GeometryShader : public Renderer::WindowApp
{
public:
    GLuint vao, vbo;

    GeometryShader(int width = 1920, int height = 1080, const std::string& title = "GeometryShader", const std::string& cameraType = "tps")
        : WindowApp(width, height, title, cameraType)
    {
    }

    ~GeometryShader() = default;

    virtual void Init() override
    {
        float points[] =
        {
            -0.5, 0.5f, 1.0f, 0.0, 0.0,
            0.5f, 0.5f, 0.0, 1.0, 0.0,
            0.5f, -0.5f, 0.0, 0.0, 1.0,
            -0.5f, -0.5f, 1.0, 1.0, 0.0
        };
        myShader = m_shaderManager->loadShader("geometryTest", SHADER_PATH"/GeometryTest/GeometryTest.vs", SHADER_PATH"/GeometryTest/GeometryTest.fs", SHADER_PATH"/GeometryTest/GeometryTest.gs");

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
        // ENABLE Point sprite
        glEnable(GL_PROGRAM_POINT_SIZE);
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
        glDrawArrays(GL_POINTS, 0, 4);
        m_shaderManager->unbindShader();
        DrawImGui();
        m_renderDevice->endFrame();
    }

    void DrawImGui()
    {
        // imgui
        {
            ImGui::Begin("GeometryShader Example");
            ImGui::Text("GeometryShader Example");
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
    bool m_enableDepthTest, m_enablePolygonMode{false};
    GLuint myShader;
    glm::vec3 sunLightDir{ glm::vec3(0.072f, 0.42f, 1.0f) };
    glm::vec3 sunLightColor{ glm::vec3(0.5) };
    float ambientCoef{ 0.5f };
    float diffuseCoef{ 0.6f };
    float specularCoef{ 0.6f };
};

