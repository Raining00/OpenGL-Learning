#pragma once
#include "RenderApp/WindowApp.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "Drawable/StaticModelDrawable.h"

class GammaCorrect : public Renderer::WindowApp
{
public:
    GammaCorrect(int width = 1920, int height = 1080, const std::string& title = "GammaCorrect", const std::string& cameraType = "tps")
        : WindowApp(width, height, title, cameraType)
    {
        m_halfScreenWidth = m_renderDevice->getWindowWidth() / 2;
    }

    ~GammaCorrect() = default;

    virtual void Init() override
    {
        //shaders
        unsigned int phoneShader = m_shaderManager->loadShader("phoneShader", SHADER_PATH"/phoneLight.vs", SHADER_PATH"/phoneLight.fs");
        unsigned int blingphoneShader = m_shaderManager->loadShader("blingphoneShader", SHADER_PATH"/GammaCorrect/GammaCorrect.vs", SHADER_PATH"/GammaCorrect/GammaCorrect.fs");

        // texture
        unsigned int diffuseMap = m_textureManager->loadTexture2D("diffuseMap", ASSETS_PATH"/texture/floor.png",Renderer::TextureType::DIFFUSE);
        unsigned int specularMap = m_textureManager->loadTexture2D("specular", ASSETS_PATH"/texture/floor.png",Renderer::TextureType::SPECULAR);
        float scale = 50.f;
        unsigned int floor = m_meshManager->loadMesh(new Renderer::Plane(1.0, 1.0, scale));
        //m_renderSystem->createSkyBox(ASSETS_PATH  "/texture/skybox/", ".jpg");
        glm::vec3 ambient = glm::vec3(0.2f, 0.1f, 0.05f); 
        glm::vec3 diffuse = glm::vec3(0.8f, 0.4f, 0.2f); 
        glm::vec3 specular = glm::vec3(1.0f, 0.5f, 0.3f); 

        //m_renderSystem->setSunLight(glm::vec3(1.0f, 0.5f, -0.5f), ambient, diffuse, specular);
        //m_renderSystem->createSunLightCamera(glm::vec3(0.0f), -600.0f, +600.0f,
        //    -600.0f, +600.0f, 1.0f, 500.0f);

        m_lightManager->CreatePointLight("PointLight0", glm::vec3(0.0f, 0.3f, 0.f), ambient, diffuse, specular, 1.0f, 0.7, 1.8);
        m_lightManager->CreatePointLight("PointLight1", glm::vec3(7.f, 0.3f, 0.f), ambient, diffuse, specular, 1.0f, 0.7, 1.8);
        m_lightManager->CreatePointLight("PointLight2", glm::vec3(-7.f, 0.3f, 0.f), ambient, diffuse, specular, 1.0f, 0.7, 1.8);
        

        // add drawable
        m_renderSystem->UseDrawableList(true);
        Renderer::SimpleDrawable* contianer[1];
        // floor plan
        contianer[0] = new Renderer::SimpleDrawable(blingphoneShader);
        contianer[0]->addMesh(floor);
        contianer[0]->addTexture(diffuseMap);
        contianer[0]->addTexture(specularMap);
        contianer[0]->setReceiveShadow(false);
        contianer[0]->setProduceShadow(false);
        contianer[0]->getTransformation()->setScale(glm::vec3(scale));
        contianer[0]->getTransformation()->setTranslation(glm::vec3(0.0f, 0.0, 0.0f));
        contianer[0]->getTransformation()->setRotation(glm::vec3(0.f, 0.0f, 0.0f));
        m_renderSystem->addDrawable(contianer[0]);

        //m_renderSystem->setCullFace(false, GL_BACK);
        m_renderSystem->setBlend(true, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        m_renderSystem->createFrameBuffer(m_renderDevice->getWindowWidth(), m_renderDevice->getWindowHeight());
    }

    virtual void Render() override
    {
        m_renderSystem->setClearColor(glm::vec4(m_BackColor, 1.0f));
        //m_renderSystem->setSunLight(sunLightDir, sunLightColorAmbient, sunLightColorDiffse, sunLightColorSpecular);
        m_renderSystem->render(true);
        m_shaderManager->bindShader("blingphoneShader");
        m_shaderManager->getShader("blingphoneShader")->setBool("compareDifferent", Commpared);
        m_shaderManager->getShader("blingphoneShader")->setFloat("halfScreenWidth", m_halfScreenWidth);

        m_shaderManager->getShader("blingphoneShader")->setBool("UseBlingPhone", BlingPhoneOn);
        m_shaderManager->getShader("blingphoneShader")->setBool("GammaCorrectOn", GammaCorrectOn);
        m_shaderManager->getShader("blingphoneShader")->setFloat("gamma", m_gamma);
        m_shaderManager->unbindShader();
    }

    void RenderUI()
    {
        // imgui
        {
            ImGui::Begin("GammaCorrect Example");
            ImGui::Text("GammaCorrect Example");
            ImGui::ColorEdit3("backgroundColor", (float*)&m_BackColor);
            ImGui::Text("SunLight");
            ImGui::SliderFloat3("sunLightDir", (float*)&sunLightDir, -1.0f, 1.0f);
            ImGui::ColorEdit3("sunLightColorAmbient", (float*)&sunLightColorAmbient);
            ImGui::ColorEdit3("sunLightColorDiffse", (float*)&sunLightColorDiffse);
            ImGui::ColorEdit3("sunLightColorSpecular", (float*)&sunLightColorSpecular);
            ImGui::Checkbox("Commpared", (bool*)&Commpared);
            ImGui::Checkbox("BlingPhoneOn", (bool*) & BlingPhoneOn);
            ImGui::Checkbox("GammaCorrectOn", (bool*) &GammaCorrectOn);
            ImGui::DragFloat("Gamma Value: ", (float*)&m_gamma, 0.01, 0.f, 20.f);
            ImGui::DragFloat("halfScreenWidth: ", (float*)&m_halfScreenWidth);
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
