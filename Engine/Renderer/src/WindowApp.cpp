#include "WindowApp.h"
#include "TPSCamera.h"
#include "FPSCamera.h"
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
            m_camera->setPerspective(45.0f, static_cast<float>(m_renderDevice->getWindowWidth()) / m_renderDevice->getWindowHeight(), 0.1f, 100.0f);
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
            m_camera->setPerspective(45.0f, static_cast<float>(m_renderDevice->getWindowWidth()) / m_renderDevice->getWindowHeight(), 0.1f, 100.0f);
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

    void WindowApp::Render()
    {
        m_renderDevice->beginFrame();
        m_renderSystem->setClearColor(glm::vec4(0.0, 0.0, 0.0, 1.0f));
        m_renderSystem->render();
        {
            // Render UI
            ImGui::Begin("WindowApp");
            ImGui::Text("This is a example of WindowApp");
            ImGui::End();
        }
        m_renderDevice->endFrame();
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
            Render();
        }
        Release();
    }

}