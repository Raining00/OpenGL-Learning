#pragma once
#include "RenderApp/WindowApp.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "Drawable/StaticModelDrawable.h"
class ShadowMapping : public Renderer::WindowApp
{
public:
    ShadowMapping(int width = 1920, int height = 1080, const std::string& title = "GammaCorrect", const std::string& cameraType = "tps")
        : WindowApp(width, height, title, cameraType)
    {
        m_halfScreenWidth = m_renderDevice->getWindowWidth() / 2;
    }

    ~ShadowMapping() = default;

    virtual void Init() override
    {
        //shaders
        unsigned int phoneShader = m_shaderManager->loadShader("phoneShader", SHADER_PATH"/phoneLight.vs", SHADER_PATH"/phoneLight.fs");
        unsigned int blingphoneShader = m_shaderManager->loadShader("blingphoneShader", SHADER_PATH"/GammaCorrect/GammaCorrect.vs", SHADER_PATH"/GammaCorrect/GammaCorrect.fs");
        unsigned int SimpleShow = m_shaderManager->loadShader("SimpleShow", SHADER_PATH"/SimpleShow/SimpleShow.vs", SHADER_PATH"/SimpleShow/SimpleShow.fs");

        // texture
        unsigned int diffuseMap = m_textureManager->loadTexture2D("diffuseMap", ASSETS_PATH"/texture/floor.png",  Renderer::TextureType::DIFFUSE);
        unsigned int specularMap = m_textureManager->loadTexture2D("specularMap", ASSETS_PATH"/texture/109447235_p0.jpg", Renderer::TextureType::DIFFUSE);
        unsigned int blendMap = m_textureManager->loadTexture2D("blendMap", ASSETS_PATH"/texture/blending_transparent_window.png");
        float scale = 6.f;
        unsigned int floor = m_meshManager->loadMesh(new Renderer::Plane(1.0, 1.0, scale));
        unsigned int planeMesh = m_meshManager->loadMesh(new Renderer::Plane(1.0, 1.0));
        unsigned int cubeMesh = m_meshManager->loadMesh(new Renderer::Cube(1.0, 1.0, 1.0));
        unsigned int sphereMesh = m_meshManager->loadMesh(new Renderer::Sphere(1.0, 50, 50));
        m_renderSystem->createSkyBox(ASSETS_PATH  "/texture/skybox/", ".jpg");
        glm::vec3 ambient = glm::vec3(0.2f, 0.1f, 0.05f);
        glm::vec3 diffuse = glm::vec3(0.8f, 0.4f, 0.2f); 
        glm::vec3 specular = glm::vec3(1.0f, 0.5f, 0.3f); 

        m_renderSystem->setSunLight(sunLightDir, ambient, diffuse, specular);
        m_renderSystem->createSunLightCamera(glm::vec3(0.0f), -10.f, +10.0f,
            -10.0f, +10.0f, 1.0f, 10.f);

        // add drawable
        m_renderSystem->UseDrawableList(true);

        Renderer::StaticModelDrawable* model[2];
        model[0] = new Renderer::StaticModelDrawable(SimpleShow, "O:/scene1.obj");
        model[0]->setReceiveShadow(false);
        model[0]->setProduceShadow(true);
        model[0]->getTransformation()->setTranslation(glm::vec3(0.0f, 0.0, 0.0f));
        m_renderSystem->addDrawable(model[0]);

        model[1] = new Renderer::StaticModelDrawable(SimpleShow, "O:/soap.obj");
        model[1]->setReceiveShadow(false);
        model[1]->setProduceShadow(true);
        model[1]->getTransformation()->setTranslation(glm::vec3(0.2f, 0.3, 0.0f));
        m_renderSystem->addDrawable(model[1]);


        m_renderSystem->setCullFace(false, GL_BACK);
        m_renderSystem->setBlend(true, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        m_renderSystem->createFrameBuffer(m_renderDevice->getWindowWidth(), m_renderDevice->getWindowHeight());
        m_renderSystem->createShadowDepthBuffer(m_renderDevice->getWindowWidth(), m_renderDevice->getWindowHeight());
    }

    virtual void Render() override
    {
        m_renderSystem->useLightCamera(useLightCamera);
        m_renderSystem->setClearColor(glm::vec4(m_BackColor, 1.0f));
        m_renderSystem->setSunLight(sunLightDir, sunLightColorAmbient, sunLightColorDiffse, sunLightColorSpecular);
        m_renderSystem->render(true);
        m_shaderManager->bindShader("blingphoneShader");
        m_shaderManager->getShader("blingphoneShader")->setBool("compareDifferent", Commpared);
        m_shaderManager->getShader("blingphoneShader")->setFloat("halfScreenWidth", m_halfScreenWidth);

        m_shaderManager->getShader("blingphoneShader")->setBool("UseBlingPhone", BlingPhoneOn);
        m_shaderManager->getShader("blingphoneShader")->setBool("GammaCorrectOn", GammaCorrectOn);
        m_shaderManager->getShader("blingphoneShader")->setFloat("gamma", m_gamma);
        m_shaderManager->unbindShader();
    }

    void RenderUI() override
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
            ImGui::Checkbox("ShowShadowMap", &m_renderSystem->getShowShadowMap());
            ImGui::Checkbox("UseLightCamera", (bool*)&useLightCamera);
            if(ImGui::Button("click me"))
            {
                PRINT_CYAN_BLINK("click me");
                m_renderSystem->saveDepthFrameBuffer("O:/depth.png");
            }
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
    bool BlingPhoneOn{ true }, GammaCorrectOn{ true }, Commpared{ false }, useLightCamera{ false };
    float m_gamma{ 2.2 }, m_halfScreenWidth;
};
