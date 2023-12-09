#include "RenderSystem.h"
#include "TPSCamera.h"
#include "FPSCamera.h"
#include "Config.h"

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
        FPSCamera *_cam = new FPSCamera(pos);
        _cam->lookAt(glm::normalize(target - pos), Camera3D::LocalUp);
        _cam->createUBO();
        m_camera = std::shared_ptr<Camera3D>(_cam);
        return m_camera;
    }

    Camera3D::ptr RenderSystem::createTPSCamera(glm::vec3 pos, glm::vec3 target)
    {
        TPSCamera *_cam = new TPSCamera(target, 0.0f, 30.0f, 3.0f);
        _cam->createUBO();
        m_camera = std::shared_ptr<Camera3D>(_cam);
        return m_camera;
    }

    void RenderSystem::createSunLightCamera(glm::vec3 target, float left, float right, float bottom, float top, float near, float far)
    {
        DirectionalLight* sunLight = reinterpret_cast<DirectionalLight*>(m_lightManager->getLight("SunLight").get());
        if (sunLight == nullptr)
        {
            PRINT_WARNING("You have to set the sun light first before creating the sun light camera");
            return;
        }
        const float length = 200;
        glm::vec3 pos = length * sunLight->getDirection();
        if (m_lightCamera == nullptr)
        {
            FPSCamera *_cam = new FPSCamera(pos);
            m_lightCamera = std::shared_ptr<Camera3D>(_cam);
        }
        m_lightCamera->setOrtho(left, right, bottom, top, near, far);
        FPSCamera *_cam = static_cast<FPSCamera *>(m_lightCamera.get());
        _cam->lookAt(-sunLight->getDirection(), Camera3D::LocalUp);
    }

    void RenderSystem::createFrameBuffer(int width, int height)
    {
        m_frameBuffer = std::make_shared<FrameBuffer>(width, height, "", "", std::vector<std::string>{"Color"});

        // Framebuffer shader
        m_shaderManager->loadShader("Framebuffer", SHADER_PATH "/FrameBuffer/FrameBuffer.vs", SHADER_PATH"/FrameBuffer/FrameBuffer.fs");
        // create quad
        m_screenQuad = m_meshManager->loadMesh(new ScreenQuad());
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
        glClearColor(m_renderState.m_clearColor.r, m_renderState.m_clearColor.g, m_renderState.m_clearColor.b, m_renderState.m_clearColor.a);
        glClear(m_renderState.m_clearMask);
        m_camera->updateMatrixUBO();
        // render the skydome.
        if (m_skyBox != nullptr)
        {
            glDepthFunc(GL_LEQUAL);
            glCullFace(GL_FRONT);
            m_skyBox->render(m_camera, m_lightCamera);
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
            m_drawableList->render(m_camera, m_lightCamera);
        }
    }

    void RenderSystem::renderWithFramebuffer()
    {
        if(m_frameBuffer == nullptr)
		{
			PRINT_WARNING("You have to create the framebuffer first before using it");
			return;
		}

        // first bind the framebuffer, and draw everything as usual.
        m_frameBuffer->bind();
        glClearColor(m_renderState.m_clearColor.r, m_renderState.m_clearColor.g, m_renderState.m_clearColor.b, m_renderState.m_clearColor.a);
        glClear(m_renderState.m_clearMask);

        m_camera->updateMatrixUBO();
        // render the skydome.
        if (m_skyBox != nullptr)
        {
            glDepthFunc(GL_LEQUAL);
            glCullFace(GL_FRONT);
            m_skyBox->render(m_camera, m_lightCamera);
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
            m_drawableList->render(m_camera, m_lightCamera);
        }
        // now we bind the default framebuffer, and draw the framebuffer texture on a quad.
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        // we should disable depth test here, otherwise the quad will be discarded.
        glDisable(GL_DEPTH_TEST);
        glClearColor(m_renderState.m_clearColor.r, m_renderState.m_clearColor.g, m_renderState.m_clearColor.b, m_renderState.m_clearColor.a);
        glClear(GL_COLOR_BUFFER_BIT);
        // draw the framebuffer texture on a quad. 
        Shader::ptr framebufferShader = m_shaderManager->getShader("Framebuffer");
        framebufferShader->use();
        framebufferShader->setInt("screenTexture", 0);
        GLuint colorTex = m_frameBuffer->getColorTextureIndex(0);
        m_textureManager->bindTexture(colorTex, 0);
        m_meshManager->drawMesh(m_screenQuad, false);
    }

    void RenderSystem::render(const bool& withFramebuffer)
    {
        if (withFramebuffer)
			renderWithFramebuffer();
		else
			renderWithoutFramebuffer();
    }

}