#include "WindowApp.h"
#include "Camera/TPSCamera.h"
#include "Camera/FPSCamera.h"
namespace Renderer
{

    WindowApp::WindowApp(int width, int height, const std::string &title, const std::string &cameraType)
    {
        m_renderDevice = RenderDevice::getInstance();
        m_renderDevice->initialize(title, width, height, false);
        m_renderSystem = m_renderDevice->getRenderSystem();

        m_shaderManager = m_renderSystem->getShaderManager();
        m_textureManager = m_renderSystem->getTextureManager();
        m_lightManager = m_renderSystem->getLightManager();
        m_meshManager = m_renderSystem->getMeshManager();

        if (cameraType == "tps")
        {
            PRINT_INFO("Use TPS Camera");
            // TPS camera
            m_camera = m_renderSystem->createTPSCamera(glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.0, 0.0, 0.0));
            m_camera->setPerspective(45.0f, static_cast<float>(m_renderDevice->getWindowWidth()) / m_renderDevice->getWindowHeight(), 0.1f, 2000.f);
            Renderer::TPSCamera *tpsCamera = reinterpret_cast<Renderer::TPSCamera *>(m_camera.get());
            tpsCamera->setPitch(15.0f);
            tpsCamera->setDistance(3.0f);
            tpsCamera->setDistanceLimit(0.01f, 1000.0f);
            tpsCamera->setWheelSensitivity(1.0f);
            tpsCamera->setMouseSensitivity(0.3f);
        }
        else if (cameraType == "fps")
        {
            PRINT_INFO("Use FPS Camera");
            // FPS camera
            m_camera = m_renderSystem->createFPSCamera(glm::vec3(0.0, 1.0, 3.0), glm::vec3(0.0, 0.0, 0.0));
            m_camera->setPerspective(45.0f, static_cast<float>(m_renderDevice->getWindowWidth()) / m_renderDevice->getWindowHeight(), 0.1f, 2000.f);
            //m_camera->setOrtho(-10, 10.f, -10.f, 10.f, 1.0f, 100.f);
            Renderer::FPSCamera *fpsCamera = reinterpret_cast<Renderer::FPSCamera *>(m_camera.get());
            fpsCamera->setMouseSensitivity(0.3f);
            fpsCamera->setMoveSpeed(5.f);
        }
        else
        {
            PRINT_ERROR("Unknown camera type: " << cameraType);
            exit(-1);
        }
    }

    WindowApp::~WindowApp()
    {
    }

    void WindowApp::Init()
    {
    }

    void WindowApp::Update()
    {
    }

    void WindowApp::preRender()
    {
        m_renderDevice->beginFrame();
    }

    void WindowApp::Render()
    {
        m_renderSystem->setClearColor(glm::vec4(0.0, 0.0, 0.0, 1.0f));
        m_renderSystem->render();
    }

    void WindowApp::postRender()
    {
        {
            ImGui::Begin("HDR");
            ImGui::Text("HDR");
            ImGui::Checkbox("HDR On", (bool *)m_renderSystem->getHDRPtr());
            ImGui::Checkbox("Bloom On", (bool *)m_renderSystem->getBloomPtr());
            ImGui::Checkbox("Show Brightness", (bool *)m_renderSystem->getshowBrightNessPtr());
            ImGui::DragFloat("Exposure", (float*)m_renderSystem->getExposurePtr(), 0.01, 0.f, 20.f);
            ImGui::End();
        }
        m_renderDevice->endFrame();
    }

    void WindowApp::RenderUI()
    {
        // Render UI
        ImGui::Begin("WindowApp");
        ImGui::Text("This is a example of WindowApp");
        ImGui::End();
    }

    void WindowApp::Release()
    {
    }

    void WindowApp::Run()
    {
        Init();
        while (m_renderDevice->run())
        {
            Update();
            preRender();
            Render();
            RenderUI();
            postRender();
        }
        Release();
    }

}