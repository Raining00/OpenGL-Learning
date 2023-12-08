#pragma once

#include <vector>
#include <memory>

#include "Light.h"
#include "Geometry.h"
#include "Camera3D.h"
#include "Transform3D.h"
#include "FrameBuffer.h"

namespace Renderer
{
    typedef struct StencilHandle {
        GLenum func{ GL_ALWAYS };
        GLint ref{ 1 };
        GLuint funcMask{ 0xFF };
        GLuint stencilMask{ 0xFF };
    }StencilHandle;

    class Drawable
    {
    public:

        typedef std::shared_ptr<Drawable> ptr;

        Drawable() = default;
        virtual ~Drawable() = default;

        virtual void render(Camera3D::ptr camera, Camera3D::ptr lightCamera, Shader::ptr shader = nullptr) = 0;
        virtual void renderDepth(Shader::ptr shader, Camera3D::ptr lightCamera) = 0;

        virtual void getAABB(glm::vec3& min, glm::vec3& max) { min = glm::vec3(0.0f); max = glm::vec3(0.0f);}

        virtual void setInstance(const bool& instance, const int& instanceNum = 0, const int& instanceVBO = 0, const GLuint& shaderAttribute = 3);

        virtual void setStencil(const bool& stencil, const StencilHandle& stencilOp, GLuint stencilShaderIndex)
		{
			m_stencil = stencil;
			m_stencilOp = stencilOp;
            m_stencilShaderIndex = stencilShaderIndex;
		}

        void setVisiable(bool visiable) { m_visible = visiable; }
        bool isVisiable() { return m_visible; }

        void setProduceShadow(const bool& produceShadow) { m_produceShadow = produceShadow; }
        void setReceiveShadow(const bool& receiveShadow) { m_receiveShadow = receiveShadow; }
        void setTransformUBO(const unsigned int& ubo) { uboMatrices = ubo; }

        void addTexture(unsigned int texIndex) { m_texIndex.push_back(texIndex); }
        void addMesh(unsigned int meshIndex) { m_meshIndex.push_back(meshIndex); }

        Transform3D* getTransformation() { return &m_transformation; }

    protected:
        virtual void renderImp();

    protected:
        bool m_instance{ false };
        bool m_receiveShadow{ true };
        bool m_produceShadow{ true };
        bool m_visible{ true };
        bool m_stencil{ false };
        int m_instanceNum{ 0 };
        Transform3D m_transformation;

        unsigned int m_shaderIndex;
        unsigned int m_stencilShaderIndex;
        unsigned int uboMatrices; // (uniform buffer object) for matrices.(Transform matrix: model, view, project, normalMatrix)
        std::vector<unsigned int> m_texIndex;
        std::vector<unsigned int> m_meshIndex;

        StencilHandle m_stencilOp;
    };

    class DrawableList : public Drawable
    {
    public:
        typedef std::shared_ptr<DrawableList> ptr;

        DrawableList() = default;
        virtual ~DrawableList() = default;

        unsigned int addDrawable(Drawable* drawable)
        {
            m_drawableList.push_back(Drawable::ptr(drawable));
            return m_drawableList.size() - 1;
        }

        unsigned int addDrawable(Drawable::ptr drawable)
        {
            m_drawableList.push_back(drawable);
            return m_drawableList.size() - 1;
        }

        virtual void render(Camera3D::ptr camera, Camera3D::ptr lightCamera, Shader::ptr shader = nullptr) override
        {
            for (auto& drawable : m_drawableList)
            {
                drawable->render(camera, lightCamera, shader);
            }
        }

        virtual void renderDepth(Shader::ptr shader, Camera3D::ptr lightCamera) override
        {
            for (auto& drawable : m_drawableList)
            {
                drawable->renderDepth(shader, lightCamera);
            }
        }

    private:
        std::vector<Drawable::ptr> m_drawableList;
    };

    class SkyBox : public Drawable
    {
    public:
        typedef std::shared_ptr<SkyBox> ptr;
        SkyBox(unsigned int ShaderIndex)
        {
            m_shaderIndex = ShaderIndex;
        }

        SkyBox() = default;

        virtual ~SkyBox() = default;

        virtual void render(Camera3D::ptr camera, Camera3D::ptr lightCamera, Shader::ptr shader = nullptr) override;
        virtual void renderDepth(Shader::ptr shader, Camera3D::ptr lightCamera) override {}
    };

    class SimpleDrawable : public Drawable
    {
    public:
        SimpleDrawable(unsigned int shaderIndex)
        {
            m_shaderIndex = shaderIndex;
        }
        ~SimpleDrawable() = default;

        virtual void render(Camera3D::ptr camera, Camera3D::ptr lightCamera, Shader::ptr shader = nullptr) override;
        virtual void renderDepth(Shader::ptr shader, Camera3D::ptr lightCamera) override;
    };

    class FramebufferDrawable : public Drawable
    {
    public:
        FramebufferDrawable(unsigned int shaderIndex, unsigned int scrWidth, unsigned int scrHeight);
        ~FramebufferDrawable() = default;

        virtual void render(Camera3D::ptr camera, Camera3D::ptr lightCamera, Shader::ptr shader = nullptr) override;
		virtual void renderDepth(Shader::ptr shader, Camera3D::ptr lightCamera) override;
    private:
        FrameBuffer::ptr m_frameBuffer;
        unsigned int m_renderTarget;		// render buffer.
        unsigned int m_depthTarget;			// depth buffer.
        unsigned int m_scrWidth;
        unsigned int m_scrHeight;
    };

    class ContainerDrawable : public Drawable
    {
    public:
        ContainerDrawable(unsigned int shaderIndex)
        {
            m_shaderIndex = shaderIndex;
        }
        ~ContainerDrawable() = default;

        virtual void render(Camera3D::ptr camera, Camera3D::ptr lightCamera, Shader::ptr shader = nullptr) override;
        virtual void renderDepth(Shader::ptr shader, Camera3D::ptr lightCamera) override;
    };

} // namespace Renderer
