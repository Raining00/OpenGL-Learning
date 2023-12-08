#pragma once

#include "ShaderManager.h"
#include "TextureManager.h"
#include "LightManager.h"
#include "MeshManager.h"
#include "Drawable.h"
#include "Camera3D.h"

namespace Renderer
{
    struct RenderState
    {
    public:
        GLenum m_depthFunc;
        GLenum m_polygonMode;
        GLenum m_cullFaceMode;
        GLenum m_blendSrc;
        GLenum m_blendDst;
        glm::vec4 m_clearColor;
        GLbitfield m_clearMask;
        bool m_depthTest, m_cullFace, m_blend;

        RenderState() : m_depthFunc(GL_LESS),
            m_polygonMode(GL_FILL),
            m_cullFaceMode(GL_BACK),
            m_blendSrc(GL_SRC_ALPHA),
            m_blendDst(GL_ONE_MINUS_SRC_ALPHA),
            m_clearColor(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)),
            m_clearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT),
            m_depthTest(true),
            m_cullFace(true),
            m_blend(false)
        {
        }
    };

    class RenderSystem
    {
    public:
        typedef std::shared_ptr<RenderSystem> ptr;

        RenderSystem() = default;
        ~RenderSystem() = default;

        void resize(int width, int height);
        void initialize(int width, int height);
        void createShadowDepthMap(int width, int height);
        void createSunLightCamera(glm::vec3 target, float left, float right, float bottom, float top, float near, float far);
        void createFrameBuffer(int width, int height);
        /**
        * @brief create a skybox.
        * make sure the images are in named in the following order:
        * right, left, top, bottom, back, front
        * otherwise the skybox will not be read successfully.
        * @param path the path of the skybox texture. (should be the folder of sky box images)
        * @param format the format of the skybox images. (should be the same, eg. jpg, png.)
        */
        void createSkyBox(const std::string& path, const std::string& format);
        void createUBO(unsigned int& uboIdx);
        Camera3D::ptr createTPSCamera(glm::vec3 pos, glm::vec3 target);
        Camera3D::ptr createFPSCamera(glm::vec3 pos, glm::vec3 target);
        void addDrawable(Drawable::ptr drawable) { m_drawableList->addDrawable(drawable); }
        void addDrawable(Drawable* drawable) { m_drawableList->addDrawable(drawable); }

        Camera3D::ptr getCamera() { return m_camera; }
        ShaderManager::ptr getShaderManager() { return m_shaderManager; }
        TextureManager::ptr getTextureManager() { return m_textureManager; }
        LightManager::ptr getLightManager() { return m_lightManager; }
        MeshManager::ptr getMeshManager() { return m_meshManager; }

        // settters
        void setSunLight(glm::vec3 direction, glm::vec3 ambient, glm::vec3 diffuse, glm::vec3 specular);
        void setPolygonMode(GLenum mode);
        void setClearMask(const GLbitfield& mask);
        void setClearColor(const glm::vec4& color);
        void setCullFace(const bool& enable, const GLenum& face);
        void setDepthTest(const bool& enable, const GLenum& func);
        void setBlend(const bool& enable, const GLenum& src, const GLenum& dst);
        void UseDrawableList(const bool& use = false) { m_useDrawableList = use; }

        void render(const bool& withFramebuffer = false);
        void renderWithFramebuffer();
        void renderWithoutFramebuffer();

        

    private:
        bool m_glowBlurEnable;
        int m_width, m_height;
        unsigned int m_screenQuad;
        RenderState m_renderState;
        SkyBox::ptr m_skyBox;

        Camera3D::ptr m_camera;
        Camera3D::ptr m_lightCamera;

        ShaderManager::ptr m_shaderManager;
        TextureManager::ptr m_textureManager;
        LightManager::ptr m_lightManager;
        MeshManager::ptr m_meshManager;

        DrawableList::ptr m_drawableList;
        bool m_useDrawableList{ false };

        FrameBuffer::ptr m_frameBuffer{ nullptr };
    private:
        void renderShadowDepth();
        void renderMotionBlurQuad();
    };

}