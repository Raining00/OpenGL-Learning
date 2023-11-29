#include "Drawable.h"

#include "TextureManager.h"
#include "ShaderManager.h"
#include "MeshManager.h"
#include "LightManager.h"

#include "ColorfulPrint.h"

namespace Renderer
{

    void Drawable::renderImp()
    {
        MeshManager::ptr meshManager = MeshManager::getSingleton();
        TextureManager::ptr textureManager = TextureManager::getSingleton();
        for (int x = 0; x < m_meshIndex.size(); x++)
        {
            if (x < m_texIndex.size())
                textureManager->bindTexture(m_texIndex[x], 0);
            else
                textureManager->unbindTexture(m_texIndex[x]);
            meshManager->drawMesh(m_meshIndex[x], m_instance, m_instanceNum);
        }
    }

    void SkyBox::render(Camera3D::ptr camera, Light::ptr sunLight, Camera3D::ptr lightCamera, Shader::ptr shader)
    {
        if (!m_visible)
            return;
        if (shader == nullptr)
            shader = ShaderManager::getSingleton()->getShader(m_shaderIndex);
        shader->use();
        shader->setInt("image", 0);
        shader->setBool("receiveShadow", m_receiveShadow);
        shader->setMat4("view", glm::mat4(glm::mat3(camera->getViewMatrix())));
        shader->setMat4("projection", camera->getProjectionMatrix());
        this->renderImp();
        ShaderManager::getSingleton()->unbindShader();
    }

    void SimpleDrawable::render(Camera3D::ptr camera, Light::ptr sunLight, Camera3D::ptr lightCamera, std::shared_ptr<Shader> shader)
    {
        if (!m_visible)
            return;
        if (shader == nullptr)
            shader = ShaderManager::getSingleton()->getShader(m_shaderIndex);
        shader->use();
        if (sunLight)
            sunLight->setLightUniforms(shader, camera, "sunLight");

        // texture
        shader->setInt("material.diffuse", 0);
        // depth map.
        Texture::ptr depthMap = TextureManager::getSingleton()->getTexture("shadowDepth");
        if (depthMap != nullptr)
        {
            shader->setInt("depthMap", 1);
            depthMap->bind(1);
        }
        // light space matrix.
        if (lightCamera != nullptr)
            shader->setMat4("lightSpaceMatrix",
                            lightCamera->getProjectionMatrix() * lightCamera->getViewMatrix());
        else
            shader->setMat4("lightSpaceMatrix", glm::mat4(1.0f));
        // object matrix.
        shader->setBool("instance", false);
        shader->setBool("receiveShadow", m_receiveShadow);
        shader->setMat4("model", m_transformation.getWorldMatrix());
        shader->setMat4("view", camera->getViewMatrix());
        shader->setMat4("projection", camera->getProjectionMatrix());
        shader->setMat3("normalMatrix", m_transformation.getNormalMatrix());
        this->renderImp();
        ShaderManager::getSingleton()->unbindShader();
    }

    void SimpleDrawable::renderDepth(std::shared_ptr<Shader> shader, Camera3D::ptr lightCamera)
    {
        if (!m_visible || !m_produceShadow)
            return;
        shader->use();
        shader->setBool("instance", false);
        shader->setMat4("lightSpaceMatrix",
                        lightCamera->getProjectionMatrix() * lightCamera->getViewMatrix());
        shader->setMat4("model", m_transformation.getWorldMatrix());
        this->renderImp();
        ShaderManager::getSingleton()->unbindShader();
    }

    void ContainerDrawable::render(Camera3D::ptr camera, Light::ptr sunLight,
                                   Camera3D::ptr lightCamera, std::shared_ptr<Shader> shader)
    {
        if (!m_visible)
            return;
        glCullFace(GL_FRONT);
        if (shader == nullptr)
            shader = ShaderManager::getSingleton()->getShader(m_shaderIndex);
        shader->use();
        if (sunLight)
            sunLight->setLightUniforms(shader, camera, "sunLight");
        shader->setInt("material.diffuse", 0);
        // depth map.
        Texture::ptr depthMap = TextureManager::getSingleton()->getTexture("shadowDepth");
        if (depthMap != nullptr)
        {
            shader->setInt("depthMap", 1);
            depthMap->bind(1);
        }
        // light space matrix.
        if (lightCamera != nullptr)
            shader->setMat4("lightSpaceMatrix",
                            lightCamera->getProjectionMatrix() * lightCamera->getViewMatrix());
        else
            shader->setMat4("lightSpaceMatrix", glm::mat4(1.0f));
        // object matrix.
        shader->setBool("instance", false);
        shader->setBool("receiveShadow", m_receiveShadow);
        shader->setMat4("model", m_transformation.getWorldMatrix());
        shader->setMat4("view", camera->getViewMatrix());
        shader->setMat4("projection", camera->getProjectionMatrix());
        shader->setMat3("normalMatrix", m_transformation.getNormalMatrix());
        this->renderImp();
        ShaderManager::getSingleton()->unbindShader();
        glCullFace(GL_BACK);
    }

    void ContainerDrawable::renderDepth(std::shared_ptr<Shader> shader, Camera3D::ptr lightCamera)
    {
        if (!m_visible || !m_produceShadow)
            return;
        shader->use();
        shader->setBool("instance", false);
        shader->setMat4("lightSpaceMatrix",
                        lightCamera->getProjectionMatrix() * lightCamera->getViewMatrix());
        shader->setMat4("model", m_transformation.getWorldMatrix());
        this->renderImp();
        ShaderManager::getSingleton()->unbindShader();
    }

} // namespace Renderer
