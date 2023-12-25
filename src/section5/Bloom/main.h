#pragma once
#include "RenderApp/WindowApp.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "Drawable/StaticModelDrawable.h"
#include "Drawable/LightDrawable.h"
class Bloom : public Renderer::WindowApp
{
public:
    Bloom(int width = 1920, int height = 1080, const std::string& title = "Bloom", const std::string& cameraType = "tps")
        : WindowApp(width, height, title, cameraType)
    {
        m_halfScreenWidth = m_renderDevice->getWindowWidth() / 2;
    }

    ~Bloom() = default;

    virtual void Init() override
    {
        //shaders
        unsigned int blingphoneShader = m_shaderManager->loadShader("blingphoneShader", SHADER_PATH"/phoneLight.vs", SHADER_PATH"/BlingPhone.fs");

        // texture
        unsigned int diffuseMap = m_textureManager->loadTexture2D("diffuseMap", ASSETS_PATH"/texture/brickwall/brickwall.jpg", Renderer::TextureType::DIFFUSE);
        unsigned int specularMap = m_textureManager->loadTexture2D("specular", ASSETS_PATH"/texture/brickwall/brickwall.jpg", Renderer::TextureType::SPECULAR); 
        unsigned int normalMap = m_textureManager->loadTexture2D("normal", ASSETS_PATH"/texture/brickwall/brickwall_normal.jpg", Renderer::TextureType::NORMAL);
        
        unsigned int cubeMap = m_textureManager->loadTexture2D("cubeDiffuse", ASSETS_PATH"/texture/109447235_p0.jpg", Renderer::TextureType::DIFFUSE);
        unsigned int cubeSpecular = m_textureManager->loadTexture2D("cubeSpecular", ASSETS_PATH"/texture/109447235_p0.jpg", Renderer::TextureType::SPECULAR);
        
        float scale = 50.f;
        unsigned int floor = m_meshManager->loadMesh(new Renderer::Plane(1.0, 1.0, scale));
        unsigned int cube = m_meshManager->loadMesh(new Renderer::Cube(1.0, 1.0, 1.0));
        unsigned int sphere = m_meshManager->loadMesh(new Renderer::Sphere(0.2, 20, 20));
        //m_renderSystem->createSkyBox(ASSETS_PATH  "/texture/skybox/", ".jpg");
        glm::vec3 ambient = glm::vec3(0.2f, 0.1f, 0.05f);
        glm::vec3 diffuse = glm::vec3(0.8f, 0.4f, 0.2f);
        glm::vec3 specular = glm::vec3(1.0f, 0.5f, 0.3f);

        //m_renderSystem->setSunLight(glm::vec3(1.0f, 0.5f, -0.5f), ambient, diffuse, specular);
        //m_renderSystem->createSunLightCamera(glm::vec3(0.0f), -600.0f, +600.0f,
        //    -600.0f, +600.0f, 1.0f, 500.0f);

        Renderer::LightDrawable* lightDrawable[3];
        lightDrawable[0] = new Renderer::LightDrawable("PointLight0", glm::vec3(0.0f, 0.5f, 0.f), ambient, diffuse * 5.f, specular * 5.f);
        lightDrawable[0]->addMesh(sphere);
        lightDrawable[0]->setDisplayColor(diffuse * 5.f);
        m_renderSystem->addDrawable(lightDrawable[0]);
        lightDrawable[1] = new Renderer::LightDrawable("PointLight1", glm::vec3(7.f, 0.5f, 0.f), glm::vec3(0.1f, 0.0f, 0.0f), glm::vec3(0.4f, 0.0f, 0.0f) * 20.f, glm::vec3(0.6f, 0.0f, 0.0f) * 10.f);
        lightDrawable[1]->addMesh(sphere);
        lightDrawable[1]->setDisplayColor(glm::vec3(200.f, 0.0f, 0.0f));
        m_renderSystem->addDrawable(lightDrawable[1]);
        lightDrawable[2] = new Renderer::LightDrawable("PointLight2", glm::vec3(-7.f, 0.5f, 0.f), glm::vec3(0.0f, 0.0f, 0.2f), glm::vec3(0.0f, 0.0f, 0.5f) * 20.f, glm::vec3(0.0f, 0.0f, 0.8f) * 10.f);
        lightDrawable[2]->addMesh(sphere);
        lightDrawable[2]->setDisplayColor(glm::vec3(0.0f, 0.f, 200.f));
        m_renderSystem->addDrawable(lightDrawable[2]);

        m_renderSystem->createShadowDepthBuffer(1024, 1024, true, Renderer::TextureType::DEPTH_CUBE);

        // add drawable
        m_renderSystem->UseDrawableList(true);
        Renderer::SimpleDrawable* contianer[3];
        // floor plan
        contianer[0] = new Renderer::SimpleDrawable(blingphoneShader);
        contianer[0]->addMesh(floor);
        contianer[0]->addTexture(diffuseMap);
        contianer[0]->addTexture(specularMap);
        contianer[0]->addTexture(normalMap);
        contianer[0]->setReceiveShadow(true);
        contianer[0]->setProduceShadow(false);
        contianer[0]->getTransformation()->setScale(glm::vec3(scale));
        contianer[0]->getTransformation()->setTranslation(glm::vec3(0.0f, 0.0, 0.0f));
        contianer[0]->getTransformation()->setRotation(glm::vec3(0.f, 0.0f, 0.0f));
        m_renderSystem->addDrawable(contianer[0]);

        contianer[1] = new Renderer::SimpleDrawable(blingphoneShader);
        contianer[1]->addMesh(cube);
        contianer[1]->addTexture(cubeMap);
        contianer[1]->addTexture(cubeSpecular);
        contianer[1]->setReceiveShadow(false);
        contianer[1]->setProduceShadow(true);
        contianer[1]->getTransformation()->setScale(glm::vec3(1.0f));
        contianer[1]->getTransformation()->setTranslation(glm::vec3(1.0f, 0.5, 0.0f));
        m_renderSystem->addDrawable(contianer[1]);

        contianer[2] = new Renderer::SimpleDrawable(blingphoneShader);
        contianer[2]->addMesh(cube);
        contianer[2]->addTexture(cubeMap);
        contianer[2]->addTexture(cubeSpecular);
        contianer[2]->setReceiveShadow(false);
        contianer[2]->setProduceShadow(true);
        contianer[2]->getTransformation()->setScale(glm::vec3(1.0f));
        contianer[2]->getTransformation()->setTranslation(glm::vec3(-1.0f, 0.5, 0.0f));
        m_renderSystem->addDrawable(contianer[2]);

        //m_renderSystem->setCullFace(false, GL_BACK);
        //m_renderSystem->setBlend(true, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        m_renderSystem->setBloomOn(true);
        m_renderSystem->createFrameBuffer(m_renderDevice->getWindowWidth(), m_renderDevice->getWindowHeight(), true);
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
