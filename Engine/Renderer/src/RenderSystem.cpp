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
    if(m_camera != nullptr)
        m_camera->setAspect(static_cast<float>(m_width) / static_cast<float>(m_height));
}

void RenderSystem::initialize(int width, int height)
{
    m_width = width;
    m_height = height;
    resize(m_width, m_height);
    m_shaderManager = ShaderManager::getInstance();
    m_textureManager = TextureManager::getInstance();
}

Camera3D::ptr RenderSystem::createFPSCamera(glm::vec3 pos, glm::vec3 target)
{
    FPSCamera* _cam = new FPSCamera(pos);
    _cam->lookAt(glm::normalize(target - pos), Camera3D::LocalUp);
    m_camera = std::shared_ptr<Camera3D>(_cam);
    return m_camera;
}

Camera3D::ptr RenderSystem::createTPSCamera(glm::vec3 pos, glm::vec3 target)
{
    TPSCamera* _cam = new TPSCamera(target, 0.0f, 30.0f, 3.0f);
    m_camera = std::shared_ptr<Camera3D>(_cam);
    return m_camera;
}

void RenderSystem::setClearMask(GLbitfield mask)
{
    m_renderState.m_clearMask = mask;
}

void RenderSystem::setClearColor(glm::vec4 color)
{
    m_renderState.m_clearColor = color;
}

void RenderSystem::setCullFace(bool enable, GLenum face)
{
    m_renderState.m_cullFace = enable;
    m_renderState.m_cullFaceMode = face;
}

void RenderSystem::setDepthTest(bool enable, GLenum func)
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
    //TODO: IMPLEMENT DIRECTIONAL LIGHT
}

void RenderSystem::render()
{
    glClearColor(m_renderState.m_clearColor.r, m_renderState.m_clearColor.g, m_renderState.m_clearColor.b, m_renderState.m_clearColor.a);
    glClear(m_renderState.m_clearMask);
    glPolygonMode(GL_FRONT_AND_BACK, m_renderState.m_polygonMode);

    // cullface 
    if(m_renderState.m_cullFace)
    {
        glEnable(GL_CULL_FACE);
        glCullFace(m_renderState.m_cullFaceMode);
    }
    else
        glDisable(GL_CULL_FACE);

    // depth test
    if(m_renderState.m_depthTest)
    {
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(m_renderState.m_depthFunc);
    }
    else
        glDisable(GL_DEPTH_TEST);
}

}