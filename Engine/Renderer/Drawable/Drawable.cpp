#include "Drawable.h"

#include "Manager/TextureManager.h"
#include "Manager/ShaderManager.h"
#include "Manager/MeshManager.h"
#include "Manager/LightManager.h"

#include "../../include/ColorfulPrint.h"

namespace Renderer
{
    void Drawable::renderImp()
    {
        MeshManager::ptr meshManager = MeshManager::getSingleton();
        TextureManager::ptr textureManager = TextureManager::getSingleton();
        for (int x = 0; x < m_meshIndex.size(); x++)
        {
            bool useNormalMap = false;
            for (int i = 0; i < m_texIndex.size(); i++)
            {
                Texture::ptr _texture = textureManager->getTexture(m_texIndex[i]);
                if (_texture->getTextureType() == TextureType::DIFFUSE)
                    _texture->bind(0);
                else if (_texture->getTextureType() == TextureType::SPECULAR)
                    _texture->bind(1);
                else if (_texture->getTextureType() == TextureType::NORMAL)
                {
                    _texture->bind(2);
                    useNormalMap = true;
                }
				else if (_texture->getTextureType() == TextureType::HEIGHT)
					_texture->bind(3);
                else if(_texture->getTextureType() == TextureType::REFLECT)
                    _texture->bind(4);
                else 
                    _texture->bind(0);
            }
            ShaderManager::getInstance()->getShader(m_shaderIndex)->setBool("material.useNormalMap", useNormalMap);
            meshManager->drawMesh(m_meshIndex[x], m_instance, m_instanceNum);
            textureManager->unbindTexture(m_texIndex[x]);
        }
    }

    void Drawable::setInstance(const bool& instance, const int& instanceNum, const int& instanceVBO, const GLuint& shaderAttribute)
    {
        if (shaderAttribute < 3)
        {
            PRINT_ERROR("shaderAttribute must be greater than 4. As the first there attributes are for vertex position, normal and texture coordinate.");
            return;
        }
        m_instance = instance;
        m_instanceNum = instanceNum;
        auto meshMgr = MeshManager::getSingleton();
        m_instanceVBO = instanceVBO;
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        for (int i = 0; i < m_meshIndex.size(); i++)
        {
            GLuint meshVAO = meshMgr->getMesh(m_meshIndex[i])->getVAO();
            glBindVertexArray(meshVAO);
            // set attribute pointers for matrix (mat4) (4 times vec4).
            GLsizei vec4Size = sizeof(glm::vec4);
            glEnableVertexAttribArray(shaderAttribute + 0);
            glVertexAttribPointer(shaderAttribute + 0, 4, GL_FLOAT, GL_FALSE, vec4Size * 4, (void*)0);
            glEnableVertexAttribArray(shaderAttribute + 1);
            glVertexAttribPointer(shaderAttribute + 1, 4, GL_FLOAT, GL_FALSE, vec4Size * 4, (void*)(sizeof(glm::vec4)));
            glEnableVertexAttribArray(shaderAttribute + 2);
            glVertexAttribPointer(shaderAttribute + 2, 4, GL_FLOAT, GL_FALSE, vec4Size * 4, (void*)(2 * sizeof(glm::vec4)));
            glEnableVertexAttribArray(shaderAttribute + 3);
            glVertexAttribPointer(shaderAttribute + 3, 4, GL_FLOAT, GL_FALSE, vec4Size * 4, (void*)(3 * sizeof(glm::vec4)));

            glVertexAttribDivisor(shaderAttribute + 0, 1);
            glVertexAttribDivisor(shaderAttribute + 1, 1);
            glVertexAttribDivisor(shaderAttribute + 2, 1);
            glVertexAttribDivisor(shaderAttribute + 3, 1);
        }
    }

    Drawable::~Drawable()
    {
        if (m_instance)
        {
            glDeleteBuffers(1, &m_instanceVBO);
        }
    }

    void SkyBox::render(Camera3D::ptr camera, Camera3D::ptr lightCamera, Shader::ptr shader)
    {
        if (!m_visible)
            return;
        if (shader == nullptr)
            shader = ShaderManager::getSingleton()->getShader(m_shaderIndex);
        shader->use();
        shader->setInt("image", 0);
        shader->setBool("receiveShadow", m_receiveShadow);
        this->renderImp();
        ShaderManager::getSingleton()->unbindShader();
    }

    void SkyBox::renderImp()
    {
        MeshManager::ptr meshManager = MeshManager::getSingleton();
        TextureManager::ptr textureManager = TextureManager::getSingleton();
        for (int x = 0; x < m_meshIndex.size(); x++)
        {
            for (int i = 0; i < m_texIndex.size(); i++)
                textureManager->bindTexture(m_texIndex[x], i);
            meshManager->drawMesh(m_meshIndex[x], m_instance, m_instanceNum);
        }
        if (m_texIndex.size() > 0)
            textureManager->unbindTexture(m_texIndex[0]);
    }

    void SimpleDrawable::render(Camera3D::ptr camera, Camera3D::ptr lightCamera, std::shared_ptr<Shader> shader)
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
        LightManager::getInstance()->setLight(shader, camera);
        // texture
        shader->setInt("material.diffuse", 0);
        shader->setInt("material.specular", 1);
        shader->setInt("material.normal", 2);
		shader->setInt("material.height", 3);
		shader->setInt("material.reflect", 4);
        // depth map.
        Texture::ptr depthMap = TextureManager::getSingleton()->getTexture("shadowDepth");
        if (depthMap != nullptr)
        {
            shader->setInt("shadowMap", 5);
            depthMap->bind(5);
        }
        Texture::ptr shadowDepthCubeMap = TextureManager::getSingleton()->getTexture("shadowDepthCube");
        if (shadowDepthCubeMap != nullptr)
        {
			shader->setInt("shadowDepthCube", 6);
			shadowDepthCubeMap->bind(6);
		}
        // light space matrix.
        if (lightCamera != nullptr)
            shader->setMat4("lightSpaceMatrix",
                lightCamera->getProjectionMatrix() * lightCamera->getViewMatrix());
        else
            shader->setMat4("lightSpaceMatrix", glm::mat4(1.0f));
        // object matrix.
        shader->setBool("instance", m_instance);
        shader->setBool("receiveShadow", m_receiveShadow);
        shader->setMat4("modelMatrix", m_transformation.getWorldMatrix());
        shader->setMat3("normalMatrix", m_transformation.getNormalMatrix());
        shader->setFloat("far_plane", 25.0f);
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
        shader->setBool("instance", m_instance);
        shader->setMat4("lightSpaceMatrix",
            lightCamera->getProjectionMatrix() * lightCamera->getViewMatrix());
        shader->setMat4("modelMatrix", m_transformation.getWorldMatrix());
        this->renderImp();
        ShaderManager::getSingleton()->unbindShader();
    }

    void SimpleDrawable::renderDepthCube(Shader::ptr shader)
    {
        if (!m_visible || !m_produceShadow)
			return;
		shader->use();
		shader->setBool("instance", m_instance);
		shader->setMat4("modelMatrix", m_transformation.getWorldMatrix());

		this->renderImp();
		ShaderManager::getSingleton()->unbindShader();
    }

    void ContainerDrawable::render(Camera3D::ptr camera, Camera3D::ptr lightCamera, std::shared_ptr<Shader> shader)
    {
        if (!m_visible)
            return;
        glCullFace(GL_FRONT);
        if (shader == nullptr)
            shader = ShaderManager::getSingleton()->getShader(m_shaderIndex);
        shader->use();
        LightManager::getInstance()->setLight(shader, camera);
        shader->setInt("material.diffuse", 0);
        shader->setInt("material.specular", 1);
        shader->setInt("material.normal", 2);
        shader->setInt("material.height", 3);
        shader->setInt("material.reflect", 4);
        // depth map.
        Texture::ptr depthMap = TextureManager::getSingleton()->getTexture("shadowDepth");
        if (depthMap != nullptr)
        {
            shader->setInt("shadowMap", 5);
            depthMap->bind(1);
        }
        Texture::ptr shadowDepthCubeMap = TextureManager::getSingleton()->getTexture("shadowDepthCube");
        if (shadowDepthCubeMap != nullptr)
        {
            shader->setInt("shadowDepthCube", 6);
            shadowDepthCubeMap->bind(6);
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
        //shader->setMat4("viewMatrix", camera->getViewMatrix());
        //shader->setMat4("projectMatrix", camera->getProjectionMatrix());
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

    void ContainerDrawable::renderDepthCube(Shader::ptr shader)
    {
        if (!m_visible || !m_produceShadow)
            return;
        shader->use();
        shader->setBool("instance", m_instance);
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
    void FramebufferDrawable::render(Camera3D::ptr camera, Camera3D::ptr lightCamera, Shader::ptr shader)
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
