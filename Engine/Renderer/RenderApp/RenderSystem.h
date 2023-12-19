#pragma once

#include "Manager/ShaderManager.h"
#include "Manager/TextureManager.h"
#include "Manager/LightManager.h"
#include "Manager/MeshManager.h"
#include "Drawable/Drawable.h"
#include "Camera/Camera3D.h"

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
        void createSunLightCamera(const glm::vec3& target, const float& left, const float& right, const float& bottom, const float& top, const float& near, const float& far, const float& distance = 10.f);
        /**
         * @brief create a point light camera.
         * @param pos the position of the point light.
         * @param target the target of the point light.
         * @param aspect the aspect ratio of the point light camera.
         * @param near the near plane of the point light camera.
         * @param far the far plane of the point light camera.
         * @param fov the field of view of the point light camera. default is 90.f. Most of the time, we don't need to change this value. 
         * By setting this to 90 degrees we make sure the viewing field is exactly large enough to fill a single face of the cubemap such that all faces align correctly to each other at the edges.
         * 
         * @note the point light camera is a perspective camera.
         * @note the point light camera is used to render the shadow map of the point light.
        */
        void createShadowDepthBuffer(int width, int height, bool hdr = false, const TextureType& textureType = TextureType::DEPTH);
        void createFrameBuffer(int width, int height, bool hdr = false);
        /**
        * @brief create a skybox.
        * make sure the images are in named in the following order:
        * right, left, top, bottom, back, front
        * otherwise the skybox will not be read successfully.
        * @param path the path of the skybox texture. (should be the folder of sky box images)
        * @param format the format of the skybox images. (should be the same, eg. jpg, png.)
        */
        void createSkyBox(const std::string& path, const std::string& format);
        Camera3D::ptr createTPSCamera(glm::vec3 pos, glm::vec3 target);
        Camera3D::ptr createFPSCamera(glm::vec3 pos, glm::vec3 target);
        void saveDepthFrameBuffer(const std::string& path);
        void saveDepthCubeFrameBuffer(const std::string& path);
        void addDrawable(Drawable::ptr drawable) { m_drawableList->addDrawable(drawable); }
        void addDrawable(Drawable* drawable) { m_drawableList->addDrawable(drawable); }

        // =====================================  Getters ==========================================================//
        Camera3D::ptr getCamera() { return m_camera; }
        ShaderManager::ptr getShaderManager() { return m_shaderManager; }
        TextureManager::ptr getTextureManager() { return m_textureManager; }
        LightManager::ptr getLightManager() { return m_lightManager; }
        MeshManager::ptr getMeshManager() { return m_meshManager; }
        bool& getShowShadowMap() { return m_showShadowMap; }

        // settters
        void setSunLight(glm::vec3 direction, glm::vec3 ambient, glm::vec3 diffuse, glm::vec3 specular);
        void setPolygonMode(GLenum mode);
        void setClearMask(const GLbitfield& mask);
        void setClearColor(const glm::vec4& color);
        void setCullFace(const bool& enable, const GLenum& face);
        void setDepthTest(const bool& enable, const GLenum& func);
        void setBlend(const bool& enable, const GLenum& src, const GLenum& dst);
        void UseDrawableList(const bool& use = false) { m_useDrawableList = use; }
        void useLightCamera(const bool& use = false) { m_activateCamera = use ? m_lightCamera : m_camera; }

        void render(const bool& withFramebuffer = false);
        void renderWithFramebuffer();
        void renderWithoutFramebuffer();      

    private:
        void renderShadowDepth();

    private:
        bool m_glowBlurEnable, m_showShadowMap{ false };
        int m_width, m_height;
        unsigned int m_screenQuad;
        RenderState m_renderState;
        std::shared_ptr<FrameBuffer> m_shadowDepthBuffer{ nullptr }, m_shadowDepthCubeBuffer{ nullptr };

        SkyBox::ptr m_skyBox;

        Camera3D::ptr m_camera;
        Camera3D::ptr m_activateCamera;
        Camera3D::ptr m_lightCamera;
        Camera3D::ptr m_pointLightCamera;

        ShaderManager::ptr m_shaderManager;
        TextureManager::ptr m_textureManager;
        LightManager::ptr m_lightManager;
        MeshManager::ptr m_meshManager;

        DrawableList::ptr m_drawableList;
        bool m_useDrawableList{ false };

        FrameBuffer::ptr m_frameBuffer{ nullptr };
    };

}