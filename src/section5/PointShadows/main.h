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
        unsigned int diffuseMap = m_textureManager->loadTexture2D("diffuseMap", ASSETS_PATH"/texture/floor.png", Renderer::TextureType::DIFFUSE);
        unsigned int specularMap = m_textureManager->loadTexture2D("specularMap", ASSETS_PATH"/texture/109447235_p0.jpg", Renderer::TextureType::DIFFUSE);
        float scale = 50.f;
        unsigned int floor = m_meshManager->loadMesh(new Renderer::Plane(1.0, 1.0, scale));
        unsigned int cubeMesh = m_meshManager->loadMesh(new Renderer::Cube(1.0, 1.0, 1.0));
        unsigned int sphereMesh = m_meshManager->loadMesh(new Renderer::Sphere(1.0, 50, 50));
        //m_renderSystem->createSkyBox(ASSETS_PATH  "/texture/skybox/", ".jpg");
        //m_renderSystem->setSunLight(sunLightDir, sunLightColorAmbient, sunLightColorDiffse, sunLightColorSpecular);
        //m_renderSystem->createSunLightCamera(glm::vec3(0.0f), -5.f, +5.0f,    
        //    -5.0f, +5.0f, 1.0f, 15.f, 5.f);
        //m_renderSystem->createShadowDepthBuffer(m_renderDevice->getWindowWidth(), m_renderDevice->getWindowHeight());
        m_renderSystem->createShadowDepthBuffer(1024, 1024, false, Renderer::TextureType::DEPTH_CUBE);
        m_renderSystem->setBlend(true, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        m_renderSystem->createFrameBuffer(m_renderDevice->getWindowWidth(), m_renderDevice->getWindowHeight());
        //m_renderSystem->setCullFace(false, GL_BACK);

        glm::vec3 PointAmbient = glm::vec3(0.2f, 0.1f, 0.05f);
        glm::vec3 PointDiffuse = glm::vec3(0.8f, 0.4f, 0.2f);
        glm::vec3 PointSpecular = glm::vec3(1.0f, 0.5f, 0.3f); 
        Renderer::LightDrawable* light = new Renderer::LightDrawable("PointLight0", PointAmbient, PointDiffuse, PointSpecular, 1.0f, 0.09f, 0.032f, glm::vec3(0.0, 2.0, 1.0));
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
        contianer[0]->addTexture(diffuseMap);
        contianer[0]->setReceiveShadow(true);
        contianer[0]->setProduceShadow(false);
        contianer[0]->getTransformation()->setScale(glm::vec3(scale));
        contianer[0]->getTransformation()->setTranslation(glm::vec3(0.0f, 0.0, 0.0f));
        contianer[0]->getTransformation()->setRotation(glm::vec3(0.f, 0.0f, 0.0f));
        m_renderSystem->addDrawable(contianer[0]);

        // cube
        contianer[1] = new Renderer::SimpleDrawable(blingphoneShader);
        contianer[1]->addMesh(cubeMesh);
        contianer[1]->addTexture(specularMap);
        contianer[1]->setReceiveShadow(false);
        contianer[1]->setProduceShadow(true);
        contianer[1]->getTransformation()->setTranslation(glm::vec3(0.0f, 0.5, 0.0f));
        m_renderSystem->addDrawable(contianer[1]);

        // sphere
        contianer[2] = new Renderer::SimpleDrawable(blingphoneShader);
        contianer[2]->addMesh(sphereMesh);
        contianer[2]->addTexture(specularMap);
        contianer[2]->setReceiveShadow(false);
        contianer[2]->setProduceShadow(true);
        contianer[2]->getTransformation()->setTranslation(glm::vec3(-2.0f, 1.0, 0.0f));
        m_renderSystem->addDrawable(contianer[2]);

        Renderer::StaticModelDrawable *model = new Renderer::StaticModelDrawable(blingphoneShader, ASSETS_PATH "/model/furina/obj/furina_white.obj");
        model->setReceiveShadow(false);
        model->setProduceShadow(true);
        model->getTransformation()->setTranslation(glm::vec3(2.0f, 0.0, 0.0f));
        m_renderSystem->addDrawable(model);
    }

    virtual void Render() override
    {
        //m_renderSystem->setSunLight(sunLightDir, sunLightColorAmbient, sunLightColorDiffse, sunLightColorSpecular);
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
        ImGui::ColorEdit3("sunLightColorAmbient", (float*)&sunLightColorAmbient);
        ImGui::ColorEdit3("sunLightColorDiffse", (float*)&sunLightColorDiffse);
        ImGui::ColorEdit3("sunLightColorSpecular", (float*)&sunLightColorSpecular);
        if(ImGui::Button("save depth cube texture"))
			m_renderSystem->saveDepthCubeFrameBuffer("O:/depthCube");
        ImGui::End();
    }

private:
    glm::vec3 sunLightDir{ glm::vec3(0.072f, 0.42f, 1.0f) };
    glm::vec3 sunLightColorAmbient{ glm::vec3(0.02) };
    glm::vec3 sunLightColorDiffse{ glm::vec3(0.4) };
    glm::vec3 sunLightColorSpecular{ glm::vec3(1.0) };
};
