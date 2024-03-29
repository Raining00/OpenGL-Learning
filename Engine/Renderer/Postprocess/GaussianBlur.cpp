#include "Postprocess/GaussianBlur.h"
#include "Manager/ShaderManager.h"
#include "Manager/MeshManager.h"
#include "Manager/TextureManager.h"
#include "Mesh/Geometry.h"
#include"include/ColorfulPrint.h"
#include "include/Config.h"

#include <sstream>

namespace Renderer
{
	GaussianBlur::GaussianBlur(int width, int height):m_width(width), m_height(height),m_blurTimes(10),m_readBuffer(0), m_writeBuffer(1)
	{
		for (int i = 0; i < 2; i++)
			m_pingPongBuffer[i] = std::make_shared<FrameBuffer>(width, height, "", "", std::vector<std::string>{std::string("GaussianBlur") + std::to_string(i)}, true);
		m_screenQuadIndex = MeshManager::getInstance()->loadMesh(new ScreenQuad());
		m_gaussianShaderIndex = ShaderManager::getInstance()->loadShader("GaussianBlur", SHADER_PATH"/FrameBuffer/FrameBuffer.vs", SHADER_PATH"/FrameBuffer/GaussBlur.fs");
	}

	void GaussianBlur::renderGaussianBlurEffect(const GLuint& blurTexIndex)
	{
        // bind brightness texture to gauss shader.
        Shader::ptr gaussianBlurShdaer = ShaderManager::getInstance()->getShader("GaussianBlur");
        gaussianBlurShdaer->use();
        GLboolean horizontal = true, first_iteration = true;
        GLuint amount = 10;
        for (int i = 0; i < 2; i++)
        {
            m_pingPongBuffer[i]->bind();
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            m_pingPongBuffer[i]->unBind(m_width, m_height);
        }
            
        // we blur it 10 times(5 is on the horizon and 5 is on the vertic).
        for (GLuint i = 0; i < amount; i++)
        {
            m_pingPongBuffer[horizontal]->bind();
            gaussianBlurShdaer->setBool("horizontal", horizontal);
            TextureManager::getInstance()->bindTexture(first_iteration ? blurTexIndex : m_pingPongBuffer[!horizontal]->getColorTextureIndex(0));
            MeshManager::getInstance()->drawMesh(m_screenQuadIndex, false);
            horizontal = !horizontal;
            std::swap(m_readBuffer, m_writeBuffer);
            if (first_iteration)
                first_iteration = false;
            m_pingPongBuffer[horizontal]->unBind(m_width, m_height);
        }
	}
    unsigned int GaussianBlur::getSceneTexIndex() const
    {
        return m_pingPongBuffer[m_readBuffer]->getColorTextureIndex(0);
    }

    void GaussianBlur::setBlurTimes(const unsigned int& t)
    {
        m_blurTimes = t;
    }
}