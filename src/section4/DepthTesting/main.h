#pragma once
#include "RenderApp/WindowApp.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "Drawable/StaticModelDrawable.h"

class DepthTest : public Renderer::WindowApp
{
public:
    DepthTest(int width = 1920, int height = 1080, const std::string &title = "SimpleMesh", const std::string &cameraType = "tps")
        : WindowApp(width, height, title, cameraType)
    {
    }

    ~DepthTest() = default;

    virtual void Init() override
    {
        //shaders
        unsigned int phoneShader = m_shaderManager->loadShader("phoneShader", SHADER_PATH"/phoneLight.vs", SHADER_PATH"/phoneLight.fs");
        unsigned int stencilShader = m_shaderManager->loadShader("stencilShader", \
            SHADER_PATH"/StencilTest/stencil.vs", SHADER_PATH"/StencilTest/stencil.fs");

        // texture
        unsigned int diffuseMap = m_textureManager->loadTexture2D("diffuseMap", ASSETS_PATH"/texture/floor.png", Renderer::TextureType::DIFFUSE);
        unsigned int specularMap = m_textureManager->loadTexture2D("specularMap", ASSETS_PATH"/texture/109447235_p0.jpg");
        float scale = 50.f;
        unsigned int planeMesh = m_meshManager->loadMesh(new Renderer::Plane(1.0, 1.0, scale));
        unsigned int cubeMesh = m_meshManager->loadMesh(new Renderer::Cube(1.0, 1.0, 1.0));
        unsigned int sphereMesh = m_meshManager->loadMesh(new Renderer::Sphere(1.0, 50, 50));
        
        m_renderSystem->setSunLight(glm::vec3(1.0f, 0.5f, -0.5f), glm::vec3(0.5), glm::vec3(0.6f), glm::vec3(0.6));
        m_renderSystem->createSunLightCamera(glm::vec3(0.0f), -600.0f, +600.0f,
            -600.0f, +600.0f, 1.0f, 500.0f);

        // add drawable
        m_renderSystem->UseDrawableList(true);
        Renderer::SimpleDrawable* contianer[2];
        // floor plan
        contianer[0] = new Renderer::SimpleDrawable(phoneShader);
        contianer[0]->addMesh(planeMesh);
        contianer[0]->addTexture(diffuseMap);
        contianer[0]->setReceiveShadow(false);
        contianer[0]->setProduceShadow(false);
        contianer[0]->getTransformation()->setScale(glm::vec3(scale));
        contianer[0]->getTransformation()->setTranslation(glm::vec3(0.0f, 0.0, 0.0f));
        contianer[0]->getTransformation()->setRotation(glm::vec3(0.f, 0.0f, 0.0f));
        m_renderSystem->addDrawable(contianer[0]);

        Renderer::StaticModelDrawable* model[3];
        model[0] = new Renderer::StaticModelDrawable(phoneShader, ASSETS_PATH "/model/furina/obj/furina_white.obj");
        Renderer::StencilHandle stencil;
        stencil.func = GL_NOTEQUAL;
        stencil.ref = 1;
        stencil.funcMask = 0xFF;
        stencil.stencilMask = 0x00;
        //model[0]->setStencil(true, stencil, stencilShader);
        m_renderSystem->addDrawable(model[0]);

        model[1] = new Renderer::StaticModelDrawable(phoneShader, ASSETS_PATH "/model/furina/obj/seahorse.obj");
        model[1]->getTransformation()->setTranslation(glm::vec3(1.0f, 0.0f, 0.5f));
        model[1]->setStencil(true, stencil, stencilShader);
        m_renderSystem->addDrawable(model[1]);

        model[2] = new Renderer::StaticModelDrawable(phoneShader, ASSETS_PATH "/model/furina/obj/furinaHome.obj");
        model[2]->getTransformation()->setTranslation(glm::vec3(0.0f, 0.0f, 20.0f));
        m_renderSystem->addDrawable(model[2]);

        //contianer[1] = new Renderer::SimpleDrawable(phoneShader);
        //contianer[1]->addMesh(cubeMesh);
        //contianer[1]->addTexture(specularMap);
        //contianer[1]->setReceiveShadow(false);
        //contianer[1]->setProduceShadow(false);
        //Renderer::StencilHandle stencil;
        //stencil.func = GL_NOTEQUAL;
        //stencil.ref = 1;
        //stencil.funcMask = 0xFF;
        //stencil.stencilMask = 0x00;
        //contianer[1]->setStencil(true, stencil, stencilShader);
        //contianer[1]->getTransformation()->setScale(glm::vec3(1.0f));
        //contianer[1]->getTransformation()->setTranslation(glm::vec3(0.0f, -0.5f, 1.0f));
        //m_renderSystem->addDrawable(contianer[1]);

        //m_renderSystem->setCullFace(false, GL_BACK);
    }

    virtual void Render() override
    {
       m_renderSystem->setClearColor(glm::vec4(m_BackColor, 1.0f));
       m_renderSystem->setSunLight(sunLightDir, glm::vec3(ambientCoef), glm::vec3(diffuseCoef), glm::vec3(specularCoef));
       m_renderSystem->render();
    }

    void RenderUI() override
    {
        // imgui
        {
            ImGui::Begin("DepthTesting Example");
            ImGui::Text("DepthTesting Example");
            ImGui::ColorEdit3("backgroundColor", (float *)&m_BackColor);
            ImGui::Text("SunLight");
            ImGui::SliderFloat3("sunLightDir", (float *)&sunLightDir, -1.0f, 1.0f);
            ImGui::ColorEdit3("sunLightColor", (float *)&sunLightColor);
            ImGui::Text("Ambient");
            ImGui::SliderFloat("ambientColor", (float *)&ambientCoef, 0.0f, 1.0f);
            ImGui::Text("Diffuse");
            ImGui::SliderFloat("diffuseColor", (float *)&diffuseCoef, 0.0, 1.0f);
            ImGui::Text("Specular");
            ImGui::SliderFloat("specularColor", (float *)&specularCoef, 0.0, 1.0f);
            ImGui::End();
        }
    }

    void Release() override
    {
    }

private:
    glm::vec3 sunLightDir{glm::vec3(0.072f, 0.42f, 1.0f)};
    glm::vec3 sunLightColor{glm::vec3(0.5)};
    float ambientCoef{0.5f};
    float diffuseCoef{0.6f};
    float specularCoef{0.6f};
};
