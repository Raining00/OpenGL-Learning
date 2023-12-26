#pragma once
#include "RenderApp/WindowApp.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "Drawable/StaticModelDrawable.h"
#include "Drawable/ParticleDrawable.h"
#include "ParticleSystem/ParticleSystem.h"
inline float frand()
{
    return rand() / (float)RAND_MAX;
}
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
        m_renderSystem->setBlend(true, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        m_renderSystem->setBloomOn(true);
        m_renderSystem->createFrameBuffer(m_renderDevice->getWindowWidth(), m_renderDevice->getWindowHeight(), true);
        m_renderSystem->setSunLight(sunLightDir, sunLightColorAmbient, sunLightColorDiffse, sunLightColorSpecular);
        m_renderSystem->UseDrawableList(true);

        float radius = 0.3f;
        Physics::ParticleSystem::ptr simpleParticles = std::make_shared<Physics::ParticleSystem>();
        unsigned int numParticles = 0;
        float spacing = radius * 2.0f;
        std::vector<glm::vec4> positions;
        std::vector<glm::vec4> velocities;
        float jitter = radius * 0.01f;
        srand(1973);

        // bottom fluid.
        glm::vec3 bottomFluidSize = glm::vec3(30.0f, 30.0f, 30.0f);
        glm::ivec3 bottomFluidDim = glm::ivec3(bottomFluidSize.x / spacing,
            bottomFluidSize.y / spacing, bottomFluidSize.z / spacing);
        for (int z = 0; z < bottomFluidDim.z; ++z)
        {
            for (int y = 0; y < bottomFluidDim.y; ++y)
            {
                for (int x = 0; x < bottomFluidDim.x; ++x)
                {
                    glm::vec4 tmp_pos, tmp_vel(0.0f, 0.0f, 0.0f, 0.0f);
                    tmp_pos.x = spacing * x + radius - 0.5f * 80.0f + (frand() * 2.0f - 1.0f) * jitter;
                    tmp_pos.y = spacing * y + radius - 0.5f * 40.0f + (frand() * 2.0f - 1.0f) * jitter;
                    tmp_pos.z = spacing * z + radius - 0.5f * 40.0f + (frand() * 2.0f - 1.0f) * jitter;
                    tmp_pos.w = 1.0;
                    positions.push_back(tmp_pos);

                    velocities.push_back(tmp_vel);
                }
            }
        }

        numParticles = positions.size();
        simpleParticles->setParticlePosition(positions);
        simpleParticles->setParticleVelocity(velocities);
        Renderer::ParticlePointSpriteDrawable* particleDrawable = new Renderer::ParticlePointSpriteDrawable(4);
        particleDrawable->setParticleRadius(radius);
        particleDrawable->setParticleVBO(simpleParticles->getPositionVBO(), positions.size());
        particleDrawable->setBaseColor(glm::vec3(1.0f, 0.6f, 0.3f));
        m_renderSystem->addDrawable(particleDrawable);
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
