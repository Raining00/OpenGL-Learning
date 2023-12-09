#pragma once
#include "WindowApp.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "StaticModelDrawable.h"

class Planet : public Renderer::WindowApp
{
public:
    Planet(int width = 1920, int height = 1080, const std::string& title = "Planet", const std::string& cameraType = "tps")
        : WindowApp(width, height, title, cameraType)
    {
    }

    ~Planet() = default;

    virtual void Init() override
    {
        //shaders
        unsigned int RockShader = m_shaderManager->loadShader("Rock", SHADER_PATH"/Planet/Rock.vs", SHADER_PATH"/Planet/Rock.fs");
        unsigned int PlanetShader = m_shaderManager->loadShader("Planet", SHADER_PATH"/Planet/Planet.vs", SHADER_PATH"/Planet/Planet.fs");
        unsigned int SunShader = m_shaderManager->loadShader("SunPlanet", SHADER_PATH"/Planet/Sun.vs", SHADER_PATH"/Planet/Sun.fs");

        // texture
        unsigned int diffuseMap = m_textureManager->loadTexture2D("diffuseMap", ASSETS_PATH"/texture/floor.png");
        unsigned int sunTexture = m_textureManager->loadTexture2D("sunTexture", ASSETS_PATH"/texture/sun.jpg");

        float scale = 50.f;
        unsigned int sphereMesh = m_meshManager->loadMesh(new Renderer::Sphere(10.0, 50, 50));
        m_renderSystem->setSunLight(glm::vec3(1.0f, 0.5f, -0.5f), glm::vec3(0.5), glm::vec3(0.6f), glm::vec3(0.6));
        m_renderSystem->createSunLightCamera(glm::vec3(0.0f), -600.0f, +600.0f,
            -600.0f, +600.0f, 1.0f, 500.0f);

        // add drawable
        m_renderSystem->UseDrawableList(true);
        Renderer::SimpleDrawable* sun = new Renderer::SimpleDrawable(SunShader);
        sun->addMesh(sphereMesh);
        sun->addTexture(sunTexture);
        sun->getTransformation()->scale(glm::vec3(30.0f));
        sun->getTransformation()->translate(glm::vec3(-500.0f, 0.0f, 0.0f));
        sun->setReceiveShadow(false);
        sun->setProduceShadow(false);
        m_renderSystem->addDrawable(sun);

        Renderer::StaticModelDrawable* model[3];
        model[0] = new Renderer::StaticModelDrawable(PlanetShader, ASSETS_PATH "/model/planet/planet.obj");
        model[0]->getTransformation()->scale(glm::vec3(5.0f));
        m_renderSystem->addDrawable(model[0]);

        model[1] = new Renderer::StaticModelDrawable(RockShader, ASSETS_PATH "/model/rock/rock.obj");
        // set instance array 
        unsigned int amount = 10000;
        glm::mat4* modelMatrices;
        modelMatrices = new glm::mat4[amount];
        //std::shared_ptr<glm::mat4[]> modelMatrices = std::make_shared<glm::mat4[]>(amount); // C++ 17 feature
        srand(glfwGetTime()); // 初始化随机种子    
        float radius = 50.0;
        float offset = 10.f;
        for (unsigned int i = 0; i < amount; i++)
        {
            glm::mat4 model(1.0f);
            // 1. 位移：分布在半径为 'radius' 的圆形上，偏移的范围是 [-offset, offset]
            float angle = (float)i / (float)amount * 360.0f;
            float displacement = (rand() % (int)(2 * offset * 100)) / 100.0f - offset;
            float x = sin(angle) * radius + displacement;
            displacement = (rand() % (int)(2 * offset * 100)) / 100.0f - offset;
            float y = displacement * 0.4f; // 让行星带的高度比x和z的宽度要小
            displacement = (rand() % (int)(2 * offset * 100)) / 100.0f - offset;
            float z = cos(angle) * radius + displacement;
            model = glm::translate(model, glm::vec3(x, y, z));

            // 2. 缩放：在 0.05 和 0.25f 之间缩放
            float scale = (rand() % 20) / 100.0f + 0.05;
            model = glm::scale(model, glm::vec3(scale));

            // 3. 旋转：绕着一个（半）随机选择的旋转轴向量进行随机的旋转
            float rotAngle = (rand() % 360);
            model = glm::rotate(model, rotAngle, glm::vec3(0.4f, 0.6f, 0.8f));

            // 4. 添加到矩阵的数组中
            modelMatrices[i] = model;
        }
        unsigned int buffer;
        glGenBuffers(1, &buffer);
        glBindBuffer(GL_ARRAY_BUFFER, buffer);
        glBufferData(GL_ARRAY_BUFFER, amount * sizeof(glm::mat4), &modelMatrices[0], GL_STATIC_DRAW);
        model[1]->setInstance(true, amount, buffer);
        m_renderSystem->addDrawable(model[1]);
        delete[] modelMatrices;

        //m_renderSystem->setCullFace(false, GL_BACK);
        m_renderSystem->setBlend(true, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        m_renderSystem->createFrameBuffer(m_renderDevice->getWindowWidth(), m_renderDevice->getWindowHeight());
    }

    virtual void Render() override
    {
        m_renderDevice->beginFrame();
        m_renderSystem->setClearColor(glm::vec4(m_BackColor, 1.0f));
        m_renderSystem->setSunLight(sunLightDir, sunLightColorAmbient, sunLightColorDiffse, sunLightColorSpecular);
        m_renderSystem->render(true);
        DrawImGui();
        m_renderDevice->endFrame();
    }

    void DrawImGui()
    {
        // imgui
        {
            ImGui::Begin("Planet Example");
            ImGui::Text("Planet Example");
            ImGui::ColorEdit3("backgroundColor", (float*)&m_BackColor);
            ImGui::Text("SunLight");
            ImGui::SliderFloat3("sunLightDir", (float*)&sunLightDir, -1.0f, 1.0f);
            ImGui::ColorEdit3("sunLightColorAmbient", (float*)&sunLightColorAmbient);
            ImGui::ColorEdit3("sunLightColorDiffse", (float*)&sunLightColorDiffse);
            ImGui::ColorEdit3("sunLightColorSpecular", (float*)&sunLightColorSpecular);
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
