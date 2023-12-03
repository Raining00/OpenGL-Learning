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
        shader->setMat4("viewMatrix", glm::mat4(glm::mat3(camera->getViewMatrix())));
        shader->setMat4("projectMatrix", camera->getProjectionMatrix());
        this->renderImp();
        ShaderManager::getSingleton()->unbindShader();
    }

    void SimpleDrawable::render(Camera3D::ptr camera, Light::ptr sunLight, Camera3D::ptr lightCamera, std::shared_ptr<Shader> shader)
    {
        if (!m_visible)
            return;
        // disable stencil test.
        glStencilMask(0x00);
        if (m_stencil)
        {
            glEnable(GL_STENCIL_TEST);
            glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
            glStencilFunc(GL_ALWAYS, 1, 0xFF);
            glStencilMask(0xFF);
        }
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
        shader->setMat4("modelMatrix", m_transformation.getWorldMatrix());
        shader->setMat4("viewMatrix", camera->getViewMatrix());
        shader->setMat4("projectMatrix", camera->getProjectionMatrix());
        shader->setMat3("normalMatrix", m_transformation.getNormalMatrix());
        this->renderImp();
        if (m_stencil)
        {
            glStencilFunc(m_stencilOp.func, m_stencilOp.ref, m_stencilOp.funcMask);
            glStencilMask(m_stencilOp.stencilMask);
            glDisable(GL_DEPTH_TEST);
            //Shader::ptr shader = ShaderManager::getInstance()->getShader("stencil");
            shader = ShaderManager::getInstance()->getShader(m_stencilShaderIndex);
            shader->use();
            float scale = 1.1f;
            glm::mat4 modelMatrix(1.0f);
            modelMatrix = glm::translate(modelMatrix, m_transformation.translation());
            modelMatrix = glm::scale(modelMatrix, glm::vec3(scale, scale, scale));
            shader->setMat4("modelMatrix", modelMatrix);
            shader->setMat4("viewMatrix", camera->getViewMatrix());
            shader->setMat4("projectMatrix", camera->getProjectionMatrix());
            shader->setMat3("normalMatrix", m_transformation.getNormalMatrix());
            this->renderImp();
            // reset and clear stencil buffer. make sure this will not affect other objects.
            glStencilMask(0xFF);
            glStencilFunc(GL_ALWAYS, 1, 0xFF);
            glEnable(GL_DEPTH_TEST);
            glClear(GL_STENCIL_BUFFER_BIT);
        }
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
        shader->setMat4("modelMatrix", m_transformation.getWorldMatrix());
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
        shader->setMat4("modelMatrix", m_transformation.getWorldMatrix());
        shader->setMat4("viewMatrix", camera->getViewMatrix());
        shader->setMat4("projectMatrix", camera->getProjectionMatrix());
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
        shader->setMat4("modelMatrix", m_transformation.getWorldMatrix());
        this->renderImp();
        ShaderManager::getSingleton()->unbindShader();
    }

    FramebufferDrawable::FramebufferDrawable(unsigned int shaderIndex, unsigned int scrWidth, unsigned int scrHeight):
        m_scrWidth(scrWidth), m_scrHeight(scrHeight)
    {
        m_shaderIndex = shaderIndex;
        //m_frameBuffer = std::make_shared<FrameBuffer>(new FrameBuffer(m_scrWidth, m_scrHeight, "", "", { "ColorAttachment" }));
    }
    void FramebufferDrawable::render(Camera3D::ptr camera, Light::ptr sunLight, Camera3D::ptr lightCamera, Shader::ptr shader)
    {
		if (!m_visible)
			return;
		m_frameBuffer->bind();
		glViewport(0, 0, m_scrWidth, m_scrHeight);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		if (shader == nullptr)
			shader = ShaderManager::getSingleton()->getShader(m_shaderIndex);
        // depth map.
        Texture::ptr depthMap = TextureManager::getSingleton()->getTexture("shadowDepth");
        if (depthMap != nullptr)
        {
            shader->setInt("depthMap", 1);
            depthMap->bind(1);
        }
        if (lightCamera != nullptr)
            shader->setMat4("lightSpaceMatrix",
                lightCamera->getProjectionMatrix() * lightCamera->getViewMatrix());
        else
            shader->setMat4("lightSpaceMatrix", glm::mat4(1.0f));
		shader->use();
		shader->setInt("material.diffuse", 0);
		shader->setBool("receiveShadow", m_receiveShadow);
		shader->setMat4("viewMatrix", glm::mat4(glm::mat3(camera->getViewMatrix())));
		shader->setMat4("projectMatrix", camera->getProjectionMatrix());
		this->renderImp();
		ShaderManager::getSingleton()->unbindShader();
		m_frameBuffer->unBind();
    }
    void FramebufferDrawable::renderDepth(std::shared_ptr<Shader> shader, Camera3D::ptr lightCamera)
    {
        if (!m_visible || !m_produceShadow)
			return;
		shader->use();
		shader->setBool("instance", false);
		shader->setMat4("lightSpaceMatrix",
			lightCamera->getProjectionMatrix() * lightCamera->getViewMatrix());
		shader->setMat4("modelMatrix", m_transformation.getWorldMatrix());
		this->renderImp();
		ShaderManager::getSingleton()->unbindShader();
    }
} // namespace Renderer
