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
        //shaders
        unsigned int phoneShader = m_shaderManager->loadShader("phoneShader", SHADER_PATH"/phoneLight.vs", SHADER_PATH"/phoneLight.fs");
        // texture
        unsigned int diffuseMap = m_textureManager->loadTexture2D("diffuseMap", ASSETS_PATH"/texture/93447255_p0.png");
        unsigned int specularMap = m_textureManager->loadTexture2D("specularMap", ASSETS_PATH"/texture/109447235_p0.jpg");
        unsigned int planeMesh = m_meshManager->loadMesh(new Renderer::Plane(1.0, 1.0));
        m_renderSystem->setSunLight(glm::vec3(1.0f, 0.5f, -0.5f), glm::vec3(0.2f), glm::vec3(0.5f), glm::vec3(1.0f));
        m_renderSystem->createSunLightCamera(glm::vec3(0.0f), -600.0f, 600.0f, -600.0f, 600.0f, 1.0f, 500.0f);

        m_renderSystem->UseDrawableList(true);
        Renderer::SimpleDrawable* contianer[1];
        contianer[0] = new Renderer::SimpleDrawable(phoneShader);
        contianer[0]->addMesh(planeMesh);
        contianer[0]->addTexture(diffuseMap);
        contianer[0]->setReceiveShadow(false);
        contianer[0]->setProduceShadow(false);
        // set as floor.
        contianer[0]->getTransformation()->setScale(glm::vec3(10.0f, 10.0f, 10.0f));
        contianer[0]->getTransformation()->setTranslation(glm::vec3(0.0f, -0.5f, 0.0f));
        contianer[0]->getTransformation()->setRotation(glm::vec3(-90.0f, 0.0f, 0.0f));
        m_renderSystem->addDrawable(contianer[0]);
    }

    virtual void Render() override
    {
       m_renderDevice->beginFrame();
       m_renderSystem->setClearColor(glm::vec4(m_BackColor, 1.0f));
       m_renderSystem->render();
       DrawImGui();
       m_renderDevice->endFrame();
    }

    void DrawImGui()
    {
        // imgui
        {
            ImGui::Begin("SimpleMesh Example");
            ImGui::Text("SimpleMesh Example");
            ImGui::ColorEdit3("backgroundColor", (float *)&m_BackColor);
            ImGui::End();
        }
    }

    void Release() override
    {
    }

private:
};
