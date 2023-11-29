#include "RenderSystem.h"
#include "TPSCamera.h"
#include "FPSCamera.h"
namespace Renderer
{

    void RenderSystem::resize(int width, int height)
    {
        m_width = width;
        m_height = height;
        glViewport(0, 0, m_width, m_height);
        if (m_camera != nullptr)
            m_camera->setAspect(static_cast<float>(m_width) / static_cast<float>(m_height));
    }

    void RenderSystem::initialize(int width, int height)
    {
        m_width = width;
        m_height = height;

        resize(m_width, m_height);
        m_shaderManager = ShaderManager::getInstance();
        m_textureManager = TextureManager::getInstance();
        m_lightManager = LightManager::getInstance();
        m_meshManager = MeshManager::getInstance();
        m_drawableList = std::make_shared<DrawableList>();
    }

    Camera3D::ptr RenderSystem::createFPSCamera(glm::vec3 pos, glm::vec3 target)
    {
        FPSCamera *_cam = new FPSCamera(pos);
        _cam->lookAt(glm::normalize(target - pos), Camera3D::LocalUp);
        m_camera = std::shared_ptr<Camera3D>(_cam);
        return m_camera;
    }

    Camera3D::ptr RenderSystem::createTPSCamera(glm::vec3 pos, glm::vec3 target)
    {
        TPSCamera *_cam = new TPSCamera(target, 0.0f, 30.0f, 3.0f);
        m_camera = std::shared_ptr<Camera3D>(_cam);
        return m_camera;
    }

    void RenderSystem::createSunLightCamera(glm::vec3 target, float left, float right, float bottom, float top, float near, float far)
    {
        if (m_sunLight == nullptr)
        {
            PRINT_WARNING("You have to set the sun light first before creating the sun light camera");
            return;
        }
        const float length = 200;
        glm::vec3 pos = length * m_sunLight->getDirection();
        if (m_lightCamera == nullptr)
        {
            FPSCamera *_cam = new FPSCamera(pos);
            m_lightCamera = std::shared_ptr<Camera3D>(_cam);
        }
        m_lightCamera->setOrtho(left, right, bottom, top, near, far);
        FPSCamera *_cam = static_cast<FPSCamera *>(m_lightCamera.get());
        _cam->lookAt(-m_sunLight->getDirection(), Camera3D::LocalUp);
    }

    void RenderSystem::setClearMask(const GLbitfield &mask)
    {
        m_renderState.m_clearMask = mask;
    }

    void RenderSystem::setClearColor(const glm::vec4 &color)
    {
        m_renderState.m_clearColor = color;
    }

    void RenderSystem::setCullFace(const bool &enable, const GLenum &face)
    {
        m_renderState.m_cullFace = enable;
        m_renderState.m_cullFaceMode = face;
    }

    void RenderSystem::setDepthTest(const bool &enable, const GLenum &func)
    {
        m_renderState.m_depthTest = enable;
        m_renderState.m_depthFunc = func;
    }

    void RenderSystem::setPolygonMode(GLenum mode)
    {
        m_renderState.m_polygonMode = mode;
    }

    void RenderSystem::setSunLight(glm::vec3 direction, glm::vec3 ambient, glm::vec3 diffuse, glm::vec3 specular)
    {
        // TODO: IMPLEMENT DIRECTIONAL LIGHT
        DirectionalLight *light = new DirectionalLight();
        light->setLightDirection(direction);
        light->setLightColor(ambient, diffuse, specular);
        m_sunLight = std::shared_ptr<DirectionalLight>(light);
    }

    void RenderSystem::render()
    {
        glClearColor(m_renderState.m_clearColor.r, m_renderState.m_clearColor.g, m_renderState.m_clearColor.b, m_renderState.m_clearColor.a);
        glClear(m_renderState.m_clearMask);

        // polygon mode
        glPolygonMode(GL_FRONT_AND_BACK, m_renderState.m_polygonMode);

        // cullface
        if (m_renderState.m_cullFace)
        {
            glEnable(GL_CULL_FACE);
            glCullFace(m_renderState.m_cullFaceMode);
        }
        else
            glDisable(GL_CULL_FACE);

        // depth test
        if (m_renderState.m_depthTest)
        {
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(m_renderState.m_depthFunc);
        }
        else
            glDisable(GL_DEPTH_TEST);

        if (m_useDrawableList)
        {
            if (m_drawableList == nullptr)
                return;
            m_drawableList->render(m_camera, m_sunLight, m_lightCamera);
        }
    }

}