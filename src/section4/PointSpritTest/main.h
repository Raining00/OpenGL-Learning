#pragma once
#include "RenderApp/WindowApp.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "Drawable/ParticleDrawable.h"

class PointSpritTest : public Renderer::WindowApp
{
public:
    PointSpritTest(int width = 1920, int height = 1080, const std::string& title = "PointSpritTest", const std::string& cameraType = "tps")
        : WindowApp(width, height, title, cameraType)
    {
    }

    ~PointSpritTest() = default;

    virtual void Init() override
    {
        //shaders
        unsigned int blingphoneShader = m_shaderManager->loadShader("blingphoneShader", SHADER_PATH"/phoneLight.vs", SHADER_PATH"/BlingPhone.fs");

        // texture
        unsigned int floorTex = m_textureManager->loadTexture2D("diffuseMap", ASSETS_PATH"/texture/floor.png");

        float scale = 100.f;
        unsigned int plan = m_meshManager->loadMesh(new Renderer::Plane(1.0f, 1.0f, scale));
        m_renderSystem->setSunLight(glm::vec3(1.0f, 0.5f, -0.5f), glm::vec3(0.5), glm::vec3(0.6f), glm::vec3(0.6));
        m_renderSystem->createSunLightCamera(glm::vec3(0.0f), -5.f, +5.0f,
            -5.0f, +5.0f, 1.0f, 15.f, 5.f);
        m_renderSystem->createShadowDepthBuffer(m_renderDevice->getWindowWidth(), m_renderDevice->getWindowHeight());

        // add drawable
        m_renderSystem->UseDrawableList(true);
        Renderer::SimpleDrawable* contianer = new Renderer::SimpleDrawable(blingphoneShader);
        contianer->addMesh(plan);
        contianer->addTexture(floorTex);
        contianer->getTransformation()->setScale(glm::vec3(scale));
        contianer->setReceiveShadow(true);
        contianer->setProduceShadow(false);
        m_renderSystem->addDrawable(contianer);

        // particle
        CArray<Vec3f> _ParticlePos;
        for (float i = 0; i < 1.0f; i += 0.02f)
        {
            for (float j = 0; j < 1.0f; j += 0.02f)
            {
                for (float k = 0; k < 1.0f; k += 0.02f)
                {
					_ParticlePos.pushBack(Vec3f(i, j, k));
				}
			}
		}
        PRINT_CYAN_BLINK("Particle Count: " << _ParticlePos.size());
        Renderer::ParticlePointSpriteDrawable* particle = new Renderer::ParticlePointSpriteDrawable(_ParticlePos.size(), 0.01f, 3);
        particle->setParticlePositions(_ParticlePos);
        //particle->setBaseColor(glm::vec3(0.0f, 0.0f, 0.5f));
        particle->setProduceShadow(true);
        m_renderSystem->addDrawable(particle);

        //m_renderSystem->setCullFace(false, GL_BACK);
        m_renderSystem->setBlend(true, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        m_renderSystem->createFrameBuffer(m_renderDevice->getWindowWidth(), m_renderDevice->getWindowHeight());
    }

    virtual void Render() override
    {
        m_renderSystem->setClearColor(glm::vec4(m_BackColor, 1.0f));
        m_renderSystem->setSunLight(sunLightDir, sunLightColorAmbient, sunLightColorDiffse, sunLightColorSpecular);
        m_renderSystem->render(true);
    }

    void RenderUI() override
    {
        // imgui
        {
            ImGui::Begin("PointSprit Example");
            ImGui::Text("PointSprit Example");
            ImGui::ColorEdit3("backgroundColor", (float*)&m_BackColor);
            ImGui::Text("SunLight");
            ImGui::SliderFloat3("sunLightDir", (float*)&sunLightDir, -1.0f, 1.0f);
            ImGui::ColorEdit3("sunLightColorAmbient", (float*)&sunLightColorAmbient);
            ImGui::ColorEdit3("sunLightColorDiffse", (float*)&sunLightColorDiffse);
            ImGui::ColorEdit3("sunLightColorSpecular", (float*)&sunLightColorSpecular);
            ImGui::Checkbox("ShowShadowMap", &m_renderSystem->getShowShadowMap());
            ImGui::End();
        }
    }

    void Release() override
    {
    }

private:
    glm::vec3 sunLightDir{ glm::vec3(-1.0, 0.42f, -0.125) };
    glm::vec3 sunLightColorAmbient{ glm::vec3(0.01) };
    glm::vec3 sunLightColorDiffse{ glm::vec3(1.0) };
    glm::vec3 sunLightColorSpecular{ glm::vec3(0.6) };
};
