#pragma once
#include "RenderApp/WindowApp.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "Drawable/StaticModelDrawable.h"
#include "Drawable/LightDrawable.h"

class ShadowMapping : public Renderer::WindowApp
{
public:
    ShadowMapping(int width = 1920, int height = 1080, const std::string& title = "ShadowMapping", const std::string& cameraType = "tps")
        : WindowApp(width, height, title, cameraType)
    {
    }

    ~ShadowMapping() = default;

    virtual void Init() override
    {
        //shaders
        unsigned int phoneShader = m_shaderManager->loadShader("phoneShader", SHADER_PATH"/phoneLight.vs", SHADER_PATH"/phoneLight.fs");
        unsigned int blingphoneShader = m_shaderManager->loadShader("blingphoneShader", SHADER_PATH"/phoneLight.vs", SHADER_PATH"/BlingPhone.fs");

        // texture
        unsigned int floorDiffuse = m_textureManager->loadTexture2D("floorDiffuse", ASSETS_PATH"/texture/floor/floor.png", glm::vec4(1.0f), Renderer::TextureType::DIFFUSE);
        unsigned int floorSpecular = m_textureManager->loadTexture2D("floorSpecular", ASSETS_PATH"/texture/floor/floor.png", glm::vec4(1.0f), Renderer::TextureType::SPECULAR);
        unsigned int floorNormal = m_textureManager->loadTexture2D("floorNormal", ASSETS_PATH"/texture/floor/normal.png", glm::vec4(1.0f), Renderer::TextureType::NORMAL);

        unsigned int brickwallDiffuse = m_textureManager->loadTexture2D("brickwallDiffuse", ASSETS_PATH"/texture/brickwall/brickwall.jpg", glm::vec4(1.0f), Renderer::TextureType::DIFFUSE);
        unsigned int brickwallSpecular = m_textureManager->loadTexture2D("brickwallSpecular", ASSETS_PATH"/texture/brickwall/brickwall.jpg", glm::vec4(1.0f), Renderer::TextureType::SPECULAR);
        unsigned int brickwallNormal = m_textureManager->loadTexture2D("brickwallNormal", ASSETS_PATH"/texture/brickwall/brickwall_normal.jpg", glm::vec4(1.0f), Renderer::TextureType::NORMAL);

        float scale = 50.f;
        unsigned int floor = m_meshManager->loadMesh(new Renderer::Plane(1.0, 1.0, scale));
        unsigned int cubeMesh = m_meshManager->loadMesh(new Renderer::Cube(1.0, 1.0, 1.0));
        unsigned int sphereMesh = m_meshManager->loadMesh(new Renderer::Sphere(1.0, 50, 50));

        m_renderSystem->setBlend(true, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        m_renderSystem->createFrameBuffer(m_renderDevice->getWindowWidth(), m_renderDevice->getWindowHeight());

        glm::vec3 PointAmbient = glm::vec3(0.2f, 0.1f, 0.05f);
        glm::vec3 PointDiffuse = glm::vec3(0.8f, 0.4f, 0.2f);
        glm::vec3 PointSpecular = glm::vec3(1.0f, 0.5f, 0.3f);
        light = new Renderer::LightDrawable("PointLight0", PointAmbient, PointDiffuse, PointSpecular, 1.0f, 0.09f, 0.032f, lightPosition);
        m_renderSystem->createShadowDepthBuffer(1024, 1024, false, Renderer::TextureType::DEPTH_CUBE);
        light->addMesh(sphereMesh);
        light->getTransformation()->scale(glm::vec3(0.1f));
        light->setLightColor(glm::vec3(1.0f, 0.5f, 0.3f));
        m_renderSystem->addDrawable(light);

        // add drawable
        m_renderSystem->UseDrawableList(true);
        Renderer::SimpleDrawable* contianer[3];
        // floor plan
        contianer[0] = new Renderer::SimpleDrawable(blingphoneShader);
        contianer[0]->addMesh(floor);
        contianer[0]->addTexture(floorDiffuse);
        contianer[0]->addTexture(floorSpecular);
        contianer[0]->setReceiveShadow(true);
        contianer[0]->setProduceShadow(false);
        contianer[0]->getTransformation()->setScale(glm::vec3(scale));
        contianer[0]->getTransformation()->setTranslation(glm::vec3(0.0f, 0.0, 0.0f));
        contianer[0]->getTransformation()->setRotation(glm::vec3(0.f, 0.0f, 0.0f));
        m_renderSystem->addDrawable(contianer[0]);

        // cube
        contianer[1] = new Renderer::SimpleDrawable(blingphoneShader);
        contianer[1]->addMesh(cubeMesh);
        contianer[1]->addTexture(brickwallDiffuse);
        contianer[1]->addTexture(brickwallSpecular);
        contianer[1]->addTexture(brickwallNormal);
        contianer[1]->setReceiveShadow(false);
        contianer[1]->setProduceShadow(true);
        contianer[1]->getTransformation()->setTranslation(glm::vec3(0.0f, 0.5, 0.0f));
        m_renderSystem->addDrawable(contianer[1]);

        // sphere
        contianer[2] = new Renderer::SimpleDrawable(blingphoneShader);
        contianer[2]->addMesh(sphereMesh);
        contianer[2]->addTexture(brickwallDiffuse);
        contianer[2]->addTexture(brickwallSpecular);
        contianer[2]->addTexture(brickwallNormal);
        contianer[2]->setReceiveShadow(false);
        contianer[2]->setProduceShadow(true);
        contianer[2]->getTransformation()->setTranslation(glm::vec3(-2.0f, 1.0, 0.0f));
        m_renderSystem->addDrawable(contianer[2]);

        Renderer::StaticModelDrawable* model = new Renderer::StaticModelDrawable(blingphoneShader, ASSETS_PATH "/model/furina/obj/furina_white.obj");
        model->setReceiveShadow(false);
        model->setProduceShadow(true);
        model->getTransformation()->setTranslation(glm::vec3(2.0f, 0.0, 0.0f));
        m_renderSystem->addDrawable(model);
    }

    virtual void Render() override
    {
        light->setLightPosition(lightPosition);
        m_renderSystem->render(true);
    }

    void RenderUI() override
    {
        // imgui
        ImGui::Begin("GammaCorrect Example");
        ImGui::Text("GammaCorrect Example");
        ImGui::ColorEdit3("backgroundColor", (float*)&m_BackColor);
        ImGui::Text("SunLight");
        ImGui::SliderFloat3("sunLightDir", (float*)&sunLightDir, -1.0f, 1.0f);
        ImGui::SliderFloat3("LightPosition", (float*)&lightPosition, -5.0, 5.0);
        ImGui::ColorEdit3("sunLightColorAmbient", (float*)&sunLightColorAmbient);
        ImGui::ColorEdit3("sunLightColorDiffse", (float*)&sunLightColorDiffse);
        ImGui::ColorEdit3("sunLightColorSpecular", (float*)&sunLightColorSpecular);
        if (ImGui::Button("save depth cube texture"))
            m_renderSystem->saveDepthCubeFrameBuffer("O:/depthCube");
        ImGui::End();
    }

private:
    glm::vec3 sunLightDir{ glm::vec3(0.072f, 0.42f, 1.0f) };
    glm::vec3 sunLightColorAmbient{ glm::vec3(0.02) };
    glm::vec3 sunLightColorDiffse{ glm::vec3(0.4) };
    glm::vec3 sunLightColorSpecular{ glm::vec3(1.0) };
    glm::vec3 lightPosition{ glm::vec3(0.0, 1.0, 0.8f) };
    Renderer::LightDrawable* light;
};
