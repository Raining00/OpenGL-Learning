#include "RenderSystem.h"
#include "Camera/FPSCamera.h"
#include "Camera/TPSCamera.h"
#include "include/Config.h"
#include "include/ColorfulPrint.h"

namespace Renderer
{

    void RenderSystem::resize(int width, int height)
    {
        m_width = width;
        m_height = height;
        glViewport(0, 0, m_width, m_height);
        if (m_activateCamera != nullptr)
            m_activateCamera->setAspect(static_cast<float>(m_width) / static_cast<float>(m_height));
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

    void RenderSystem::createSkyBox(const std::string& path, const std::string& format)
    {
        if (m_skyBox != nullptr)
        {
			PRINT_WARNING("You have already created a skybox");
			return;
		}
        unsigned int skyBoxShader = m_shaderManager->loadShader("skyboxShader", SHADER_PATH"/SkyBox/SkyBox.vs", SHADER_PATH"/SkyBox/SkyBox.fs");
		m_skyBox = std::make_shared<SkyBox>(skyBoxShader);
        unsigned int skyBoxTexture = m_textureManager->loadTextureCube("skybox", path, format);
        unsigned int mesh = m_meshManager->loadMesh(new Sphere(1.0f, 10, 10));
        m_skyBox->addTexture(skyBoxTexture);
        m_skyBox->addMesh(mesh);
	}

    Camera3D::ptr RenderSystem::createFPSCamera(glm::vec3 pos, glm::vec3 target)
    {
        FPSCamera* _cam = new FPSCamera(pos);
        _cam->lookAt(glm::normalize(target - pos), Camera3D::LocalUp);
        _cam->createUBO();
        m_camera = std::shared_ptr<Camera3D>(_cam);
        m_activateCamera = m_camera;
        return m_camera;
    }

    void RenderSystem::saveDepthFrameBuffer(const std::string& path)
    {
        if (m_shadowDepthBuffer == nullptr)
        {
			PRINT_WARNING("You have to create the shadow depth buffer first before saving it");
			return;
		}
        m_shadowDepthBuffer->saveTextureToFile(path, TextureType::DEPTH);
    }

    void RenderSystem::saveDepthCubeFrameBuffer(const std::string& path)
    {
        if (m_shadowDepthCubeBuffer == nullptr)
        {
            PRINT_WARNING("You have to create the shadow depth cube buffer first before saving it");
            return;
        }
        m_shadowDepthCubeBuffer->saveDepthCubeTexture(path);
    }

    Camera3D::ptr RenderSystem::createTPSCamera(glm::vec3 pos, glm::vec3 target)
    {
        TPSCamera *_cam = new TPSCamera(target, 0.0f, 30.0f, 3.0f);
        _cam->createUBO();
        m_camera = std::shared_ptr<Camera3D>(_cam);
        m_activateCamera = m_camera;
        return m_camera;
    }

    void RenderSystem::createSunLightCamera(const glm::vec3& target, const float& left, const float& right, const float& bottom, const float& top, const float& near, const float& far, const float& distance)
    {
        DirectionalLight* sunLight = reinterpret_cast<DirectionalLight*>(m_lightManager->getLight("SunLight").get());
        if (sunLight == nullptr)
        {
            PRINT_WARNING("You have to set the sun light first before creating the sun light camera");
            return;
        }
        const float length = distance;
        glm::vec3 pos = length * sunLight->getDirection();
        if (m_lightCamera == nullptr)
        {
            FPSCamera *_cam = new FPSCamera(pos);
            m_lightCamera = std::shared_ptr<Camera3D>(_cam);
        }
        m_lightCamera->setOrtho(left, right, bottom, top, near, far);
        FPSCamera *_cam = static_cast<FPSCamera *>(m_lightCamera.get());
        _cam->lookAt(glm::normalize( - sunLight->getDirection()), Camera3D::LocalUp);
        //m_lightCamera = m_camera;
    }

    void RenderSystem::createShadowDepthBuffer(int width, int height, bool hdr, const TextureType& textureType)
    {
        if(textureType != TextureType::DEPTH && textureType != TextureType::DEPTH_CUBE)
            throw std::runtime_error("The texture type of shadow depth buffer should be DEPTH or DEPTH_CUBE");
        if (textureType == TextureType::DEPTH)
        {
            if (m_shaderManager->getShader("shadowDepth") == nullptr)
                m_shaderManager->loadShader("shadowDepth", SHADER_PATH"/shadowDepth/shadowDepth.vs", SHADER_PATH"/shadowDepth/shadowDepth.fs");
            if (m_shaderManager->getShader("FrameBufferDepth") == nullptr)
                m_shaderManager->loadShader("FramebufferDepth", SHADER_PATH"/FrameBuffer/FrameBuffer.vs", SHADER_PATH"/FrameBuffer/FrameBufferDepth.fs");
            if (m_shadowDepthBuffer == nullptr)
            {
                FrameBuffer* framebuff = new FrameBuffer(width, height, "shadowDepth", "", {}, hdr);
                m_shadowDepthBuffer = std::shared_ptr<FrameBuffer>(framebuff);
            }
        }
        else if (textureType == TextureType::DEPTH_CUBE)
        {
            if (m_shaderManager->getShader("shadowDepthCube") == nullptr)
                m_shaderManager->loadShader("shadowDepthCube", SHADER_PATH"/shadowDepth/shadowDepthCube.vs", SHADER_PATH"/shadowDepth/shadowDepthCube.fs", \
                    SHADER_PATH"/shadowDepth/shadowDepthCube.gs");
            if (m_shadowDepthCubeBuffer == nullptr)
            {
                FrameBuffer* framebuff = new FrameBuffer(1024, 1024, TextureType::DEPTH_CUBE, false);
                m_shadowDepthCubeBuffer = std::shared_ptr<FrameBuffer>(framebuff);
            }
        }
    }

    void RenderSystem::createFrameBuffer(int width, int height, bool hdr)
    {
        m_frameBuffer = std::make_shared<FrameBuffer>(width, height, "", "", std::vector<std::string>{"FragColor", "BrightColor"}, hdr);

        // Framebuffer shader
        m_shaderManager->loadShader("Framebuffer", SHADER_PATH "/FrameBuffer/FrameBuffer.vs", SHADER_PATH"/FrameBuffer/FrameBuffer.fs");
        // create quad
        m_screenQuad = m_meshManager->loadMesh(new ScreenQuad());
        m_hdr = hdr;
        if (m_BloomOn && m_hdr)
        {
            for (int i = 0; i < 2; i++)
                m_gaussBlur[i] = std::make_shared<FrameBuffer>(m_width, m_width, "", "", std::vector<std::string>{std::string("Gauss") + std::to_string(i)}, true);
            m_shaderManager->loadShader("GaussBlur", SHADER_PATH"/FrameBuffer/FrameBuffer.vs", SHADER_PATH"/FrameBuffer/GaussBlur.fs");
        }
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
    void RenderSystem::setBlend(const bool& enable, const GLenum& src, const GLenum& dst)
    {
        m_renderState.m_blend = enable;
        m_renderState.m_blendSrc = src;
        m_renderState.m_blendDst = dst;
        if (enable)
        {
            glEnable(GL_BLEND);
            glBlendFunc(src, dst);
        }
		else
			glDisable(GL_BLEND);
    }

    void RenderSystem::setPolygonMode(GLenum mode)
    {
        m_renderState.m_polygonMode = mode;
    }

    void RenderSystem::setSunLight(glm::vec3 direction, glm::vec3 ambient, glm::vec3 diffuse, glm::vec3 specular)
    {
        // TODO: IMPLEMENT DIRECTIONAL LIGHT
        DirectionalLight* sunLight = reinterpret_cast<DirectionalLight*>(m_lightManager->getLight("SunLight").get());
        if(sunLight != nullptr)
        {
            sunLight->setLightDirection(direction);
            sunLight->setLightColor(ambient, diffuse, specular);
            return;
        }
        m_lightManager->CreateDirectionalLight("SunLight", direction, ambient, diffuse, specular);
    }

    void RenderSystem::renderWithoutFramebuffer()
    {
        // render the shadow.
        {
            renderShadowDepth();
        }

        glClearColor(m_renderState.m_clearColor.r, m_renderState.m_clearColor.g, m_renderState.m_clearColor.b, m_renderState.m_clearColor.a);
        glClear(m_renderState.m_clearMask);
        m_activateCamera->updateMatrixUBO();
        // render the skydome.
        if (m_skyBox != nullptr)
        {
            glDepthFunc(GL_LEQUAL);
            glCullFace(GL_FRONT);
            m_skyBox->render(m_activateCamera, m_lightCamera);
        }

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
            m_drawableList->render(m_activateCamera, m_lightCamera);
        }
    }

    void RenderSystem::renderWithFramebuffer()
    {
        if(m_frameBuffer == nullptr)
		{
			PRINT_WARNING("You have to create the framebuffer first before using it");
			return;
		}
        // render the shadow.
        {
            renderShadowDepth();
        }
        // first bind the framebuffer, and draw everything as usual.
        m_frameBuffer->bind();
        glClearColor(m_renderState.m_clearColor.r, m_renderState.m_clearColor.g, m_renderState.m_clearColor.b, m_renderState.m_clearColor.a);
        glClear(m_renderState.m_clearMask);

        m_activateCamera->updateMatrixUBO();
        // render the skydome.
        if (m_skyBox != nullptr)
        {
            glDepthFunc(GL_LEQUAL);
            glCullFace(GL_FRONT);
            m_skyBox->render(m_activateCamera, m_lightCamera);
        }

        // depth test
        if (m_renderState.m_depthTest)
        {
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(m_renderState.m_depthFunc);
        }
        else
            glDisable(GL_DEPTH_TEST);

        // cullface
        if (m_renderState.m_cullFace)
        {
            glEnable(GL_CULL_FACE);
            glCullFace(m_renderState.m_cullFaceMode);
        }
        else
            glDisable(GL_CULL_FACE);

        if (m_useDrawableList)
        {
            if (m_drawableList == nullptr)
                return;
            m_drawableList->render(m_activateCamera, m_lightCamera);
        }

        renderFrameBuffer();
    }

    void RenderSystem::render(const bool& withFramebuffer)
    {
        if (withFramebuffer)
			renderWithFramebuffer();
		else
			renderWithoutFramebuffer();
    }

    void Renderer::RenderSystem::renderFrameBuffer()
    {
        // draw the framebuffer texture on a quad. 
        glDisable(GL_DEPTH_TEST);
        Shader::ptr framebufferShader;
        if (m_hdr && m_BloomOn)
        {
            // Make sure we are not operating on nullptr.
            if (m_gaussBlur[0] == nullptr && m_gaussBlur[1] == nullptr)
            {
                for (int i = 0; i < 2; i++)
                    m_gaussBlur[i] = std::make_shared<FrameBuffer>(m_width, m_width, "", "", std::vector<std::string>{std::string("Gauss") + std::to_string(i)}, true);
            }
            framebufferShader = m_shaderManager->getShader("GaussBlur");
            if (framebufferShader == nullptr)
            {
                m_shaderManager->loadShader("GaussBlur", SHADER_PATH"/FrameBuffer/FrameBuffer.vs", SHADER_PATH"/FrameBuffer/GaussBlur.fs");
                framebufferShader = m_shaderManager->getShader("GaussBlur");
            }
            // get brightness texture
            GLuint brightTex = m_frameBuffer->getColorTextureIndex(1);
            // bind brightness texture to gauss shader.
            framebufferShader->use();
            GLboolean horizontal = true, first_iteration = true;
            GLuint amount = 10;
            // we blur it 10 times(5 is on the horizon and 5 is on the vertic).
            for (GLuint i = 0; i < amount; i++)
            {
                m_gaussBlur[horizontal]->bind();
                framebufferShader->setBool("horizontal", horizontal);
                m_textureManager->bindTexture(first_iteration ? brightTex : m_gaussBlur[!horizontal]->getColorTextureIndex(0));
                m_meshManager->drawMesh(m_screenQuad, false);
                horizontal = !horizontal;
                if (first_iteration)
                    first_iteration = false;
                m_gaussBlur[horizontal]->unBind(m_width, m_height);
            }
        }
        // now we bind the default framebuffer, and draw the framebuffer texture on a quad.
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        if (m_showShadowMap)
        {
            framebufferShader = m_shaderManager->getShader("FramebufferDepth");
            m_textureManager->bindTexture(m_shadowDepthBuffer->getDepthTextureIndex(), 0);
        }
        else
        {
            framebufferShader = m_shaderManager->getShader("Framebuffer");
            m_textureManager->bindTexture(m_frameBuffer->getColorTextureIndex(0), 0);
        }
        framebufferShader->use();
        if (m_hdr && m_BloomOn)
        {
            framebufferShader->setInt("bloomTexture", 1);
            m_textureManager->bindTexture(m_gaussBlur[0]->getColorTextureIndex(0), 1);
        }
        framebufferShader->setInt("screenTexture", 0);
        framebufferShader->setBool("hdr", m_hdr);
        framebufferShader->setBool("bloom", true);
        framebufferShader->setFloat("exposure", m_exposure);
        m_meshManager->drawMesh(m_screenQuad, false);
    }

    void RenderSystem::renderShadowDepth()
    {
        if (m_lightCamera == nullptr && m_shadowDepthBuffer == nullptr && m_shadowDepthCubeBuffer == nullptr)
            return;
        if (m_shadowDepthBuffer != nullptr)
        {
            m_shadowDepthBuffer->bind();
            glClear(GL_DEPTH_BUFFER_BIT);
            glEnable(GL_CULL_FACE);
            glCullFace(GL_FRONT);
            glEnable(GL_DEPTH_TEST);
            m_drawableList->renderDepth(m_shaderManager->getShader("shadowDepth"), m_lightCamera);
            m_shadowDepthBuffer->unBind(m_width, m_height);
            glDisable(GL_CULL_FACE);
            glDisable(GL_DEPTH_TEST);
        }
        if (m_shadowDepthCubeBuffer != nullptr)
        {
            m_shadowDepthCubeBuffer->bind();
            glClear(GL_DEPTH_BUFFER_BIT);
            glEnable(GL_CULL_FACE);
            glEnable(GL_DEPTH_TEST);
            Shader::ptr shader = m_shaderManager->getShader("shadowDepthCube");
            shader->use();
            shader->setFloat("far_plane", 25.0);
            glm::vec3 lightPos = LightManager::getInstance()->getLight("PointLight0")->getPosition();
            shader->setVec3("lightPos", lightPos);
            std::vector<glm::mat4> shadowTransforms;
            glm::mat4 shadowProj = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 25.0f);
            shadowTransforms.push_back(shadowProj *
                glm::lookAt(lightPos, lightPos + glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
            shadowTransforms.push_back(shadowProj *
                glm::lookAt(lightPos, lightPos + glm::vec3(-1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
            shadowTransforms.push_back(shadowProj *
                glm::lookAt(lightPos, lightPos + glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0, 1.0)));
            shadowTransforms.push_back(shadowProj *
                glm::lookAt(lightPos, lightPos + glm::vec3(0.0, -1.0, 0.0), glm::vec3(0.0, 0.0, -1.0)));
            shadowTransforms.push_back(shadowProj *
                glm::lookAt(lightPos, lightPos + glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, -1.0, 0.0)));
            shadowTransforms.push_back(shadowProj *
                glm::lookAt(lightPos, lightPos + glm::vec3(0.0, 0.0, -1.0), glm::vec3(0.0, -1.0, 0.0)));
            for (unsigned int i = 0; i < 6; ++i)
                shader->setMat4("shadowMatrices[" + std::to_string(i) + "]", shadowTransforms[i]);
            m_drawableList->renderDepthCube(shader);
            m_shadowDepthCubeBuffer->unBind(m_width, m_height);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_CULL_FACE);
        }
    }
}