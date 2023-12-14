#pragma once
#include "RenderApp/WindowApp.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "Drawable/StaticModelDrawable.h"

class ExplodeObject : public Renderer::WindowApp
{
public:
    ExplodeObject(int width = 1920, int height = 1080, const std::string& title = "ExplodeObject", const std::string& cameraType = "tps")
        : WindowApp(width, height, title, cameraType)
    {
    }

    ~ExplodeObject() = default;

    virtual void Init() override
    {
        //shaders
        unsigned int phoneShader = m_shaderManager->loadShader("phoneShader", SHADER_PATH"/phoneLight.vs", SHADER_PATH"/phoneLight.fs");
        unsigned int explodeShader = m_shaderManager->loadShader("explodeShader", SHADER_PATH"/ExplodeObject/ExplodeObject.vs", SHADER_PATH"/ExplodeObject/ExplodeObject.fs", SHADER_PATH"/ExplodeObject/ExplodeObject.gs");
        unsigned int normalDisplay = m_shaderManager->loadShader("normalDisplay", SHADER_PATH"/NormalDisplay/NormalDisplay.vs", SHADER_PATH"/NormalDisplay/NormalDisplay.fs", SHADER_PATH"/NormalDisplay/NormalDisplay.gs");
        // texture
        unsigned int diffuseMap = m_textureManager->loadTexture2D("diffuseMap", ASSETS_PATH"/texture/floor.png");
        unsigned int specularMap = m_textureManager->loadTexture2D("specularMap", ASSETS_PATH"/texture/109447235_p0.jpg");
        unsigned int blendMap = m_textureManager->loadTexture2D("blendMap", ASSETS_PATH"/texture/blending_transparent_window.png");
        float scale = 50.f;
        unsigned int floor = m_meshManager->loadMesh(new Renderer::Plane(1.0, 1.0, scale));
        unsigned int planeMesh = m_meshManager->loadMesh(new Renderer::Plane(1.0, 1.0));
        unsigned int cubeMesh = m_meshManager->loadMesh(new Renderer::Cube(1.0, 1.0, 1.0));
        unsigned int sphereMesh = m_meshManager->loadMesh(new Renderer::Sphere(1.0, 50, 50));
        m_renderSystem->createSkyBox(ASSETS_PATH  "/texture/skybox/", ".jpg");
        m_renderSystem->setSunLight(glm::vec3(1.0f, 0.5f, -0.5f), glm::vec3(0.5), glm::vec3(0.6f), glm::vec3(0.6));
        m_renderSystem->createSunLightCamera(glm::vec3(0.0f), -600.0f, +600.0f,
            -600.0f, +600.0f, 1.0f, 500.0f);

        // add drawable
        m_renderSystem->UseDrawableList(true);
        Renderer::SimpleDrawable* contianer[1];
        // floor plan
        contianer[0] = new Renderer::SimpleDrawable(phoneShader);
        contianer[0]->addMesh(floor);
        contianer[0]->addTexture(diffuseMap);
        contianer[0]->setReceiveShadow(false);
        contianer[0]->setProduceShadow(false);
        contianer[0]->getTransformation()->setScale(glm::vec3(scale));
        contianer[0]->getTransformation()->setTranslation(glm::vec3(0.0f, 0.0, 0.0f));
        contianer[0]->getTransformation()->setRotation(glm::vec3(0.f, 0.0f, 0.0f));
        m_renderSystem->addDrawable(contianer[0]);

        Renderer::StaticModelDrawable* model[2];
        model[0] = new Renderer::StaticModelDrawable(phoneShader, ASSETS_PATH "/model/furina/obj/furina_white.obj");
        model[0]->getTransformation()->scale(glm::vec3(2));
        model[1]->getTransformation()->setTranslation(glm::vec3(5.0f, 0.0f, 0.5f));
        model[0]->showNormal(true, 0.05);
        m_renderSystem->addDrawable(model[0]);

        model[1] = new Renderer::StaticModelDrawable(phoneShader, ASSETS_PATH "/model/nanosuit_reflection/nanosuit.obj");
        model[1]->showNormal(true, 0.4);
        //model[1]->getTransformation()->scale(glm::vec3(0.1));
        model[1]->getTransformation()->setTranslation(glm::vec3(1.0f, 0.0f, 0.5f));
        m_renderSystem->addDrawable(model[1]);
        //m_renderSystem->setCullFace(false, GL_BACK);

        m_renderSystem->setBlend(true, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        m_renderSystem->createFrameBuffer(m_renderDevice->getWindowWidth(), m_renderDevice->getWindowHeight());
    }

    virtual void Render() override
    {
        m_renderDevice->beginFrame();
        m_renderSystem->setClearColor(glm::vec4(m_BackColor, 1.0f));
        m_renderSystem->setSunLight(sunLightDir, glm::vec3(ambientCoef), glm::vec3(diffuseCoef), glm::vec3(specularCoef));
        //m_shaderManager->getShader("explodeShader")->use();
        //m_shaderManager->getShader("explodeShader")->setFloat("sperateDistance", m_sperateDistance);
        //m_shaderManager->getShader("explodeShader")->setFloat("sperateDistance", (float)glfwGetTime());
        m_renderSystem->render(true);
        DrawImGui();
        m_renderDevice->endFrame();
    }

    void DrawImGui()
    {
        // imgui
        {
            ImGui::Begin("ExplodeObject Example");
            ImGui::Text("ExplodeObject Example");
            ImGui::ColorEdit3("backgroundColor", (float*)&m_BackColor);
            ImGui::Text("SunLight");
            ImGui::SliderFloat3("sunLightDir", (float*)&sunLightDir, -1.0f, 1.0f);
            ImGui::ColorEdit3("sunLightColor", (float*)&sunLightColor);
            ImGui::Text("Ambient");
            ImGui::SliderFloat("ambientColor", (float*)&ambientCoef, 0.0f, 1.0f);

            ImGui::DragFloat("sperateDistance", &m_sperateDistance, 0.001f, 0.0f, 1.0f);

            ImGui::Text("Diffuse");
            ImGui::SliderFloat("diffuseColor", (float*)&diffuseCoef, 0.0, 1.0f);
            ImGui::Text("Specular");
            ImGui::SliderFloat("specularColor", (float*)&specularCoef, 0.0, 1.0f);
            ImGui::End();
        }
    }

    void Release() override
    {
    }

private:
    glm::vec3 sunLightDir{ glm::vec3(0.072f, 0.42f, 1.0f) };
    glm::vec3 sunLightColor{ glm::vec3(0.5) };
    float ambientCoef{ 0.5f };
    float diffuseCoef{ 0.6f };
    float specularCoef{ 0.6f };
    float m_sperateDistance{ 0.01f };
};
