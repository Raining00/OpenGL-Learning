#pragma once
#include "RenderDevice.h"

#include "ColorfulPrint.h"
#include "Config.h"

namespace Renderer{
class WindowApp
{
public:
    WindowApp(int width=1920, int height=1080, const std::string& title="WindowApp", const std::string& cameraType="tps");
    virtual ~WindowApp();

    virtual void Init();
    virtual void Update();
    virtual void Render();
    virtual void Release();

    void Run();

protected:
    RenderDevice::ptr m_renderDevice;
    RenderSystem::ptr m_renderSystem;
    Camera3D::ptr m_camera;

    ShaderManager::ptr m_shaderManager;
    TextureManager::ptr m_textureManager;
    LightManager::ptr m_lightManager;
    MeshManager::ptr m_meshManager;
    
    DrawableList::ptr m_drawableList;
    glm::vec3 m_BackColor{ 0.0f,0.0f,0.0f };
};

}