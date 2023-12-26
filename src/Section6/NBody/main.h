#pragma once
#include "RenderApp/WindowApp.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "Drawable/StaticModelDrawable.h"
#include "Drawable/ParticleDrawable.h"

class NBody : public Renderer::WindowApp
{
public:
    NBody(int width = 1920, int height = 1080, const std::string& title = "Bloom", const std::string& cameraType = "tps")
        : WindowApp(width, height, title, cameraType)
    {
        m_halfScreenWidth = m_renderDevice->getWindowWidth() / 2;
    }

    ~NBody() = default;

    virtual void Init() override
    {
        //shaders
        unsigned int blingphoneShader = m_shaderManager->loadShader("blingphoneShader", SHADER_PATH"/phoneLight.vs", SHADER_PATH"/BlingPhone.fs");
        m_renderSystem->setCullFace(false, GL_BACK);
        m_renderSystem->setBlend(true, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        m_renderSystem->setBloomOn(true);
        m_renderSystem->createFrameBuffer(m_renderDevice->getWindowWidth(), m_renderDevice->getWindowHeight(), true);
        
        m_renderSystem->UseDrawableList(true);
        Renderer::ParticlePointSpriteDrawable* particleDrawable = new Renderer::ParticlePointSpriteDrawable(100, 0.01, 4);

    }

    virtual void Render() override
    {
        m_renderSystem->render(true);
    }

    void RenderUI()
    {
        // imgui
        {
            ImGui::Begin("Bloom Example");
            ImGui::Text("Bloom Example");
            ImGui::End();
        }
    }

    void Release() override
    {
    }

private:
    glm::vec3 sunLightDir{ glm::vec3(0.072f, 0.42f, 1.0f) };
    glm::vec3 sunLightColorAmbient{ glm::vec3(0.02) };
    glm::vec3 sunLightColorDiffse{ glm::vec3(0.4) };
    glm::vec3 sunLightColorSpecular{ glm::vec3(1.0) };
    bool BlingPhoneOn{ true }, GammaCorrectOn{ true }, Commpared{ false };
    float m_gamma{ 2.2 }, m_halfScreenWidth;
};
