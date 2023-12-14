#include "FrameBuffer.h"
#include "include/ColorfulPrint.h"
#include "Manager/TextureManager.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace Renderer
{
	FrameBuffer::FrameBuffer(int width, int height, const std::string& depthName, const std::string& stencilName,\
		const std::vector<std::string>& colorNames, bool hdr):
		m_width(width), m_height(height), m_hdr(hdr)
	{
		glGenFramebuffers(1, &m_id);

		m_colorTexIndex.resize(colorNames.size());

		for(int x = 0; x < colorNames.size(); x++)
			setupColorFramebuffer(colorNames[x], x);
		if(!depthName.empty())
			setupDepthFramebuffer(depthName);
		if(!stencilName.empty())
			setupStencilFramebuffer(stencilName);
		// Generate render buffer object
		if (depthName.empty() && stencilName.empty())
		{
			glGenRenderbuffers(1, &m_rbo);
			glBindRenderbuffer(GL_RENDERBUFFER, m_rbo);
			glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, m_width, m_height);
			glBindRenderbuffer(GL_RENDERBUFFER, 0);
			glBindFramebuffer(GL_FRAMEBUFFER, m_id);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_rbo);
		}
		glBindFramebuffer(GL_FRAMEBUFFER, m_id);
		if (colorNames.size() > 0)
		{
			glDrawBuffers(colorNames.size(), ColorAttachments);
		}
		else
		{
			glDrawBuffer(GL_NONE);
			glReadBuffer(GL_NONE);
		}
		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			PRINT_ERROR("Framebuffer not complete!");
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	FrameBuffer::FrameBuffer(int width, int height, bool hdr):
		m_width(width), m_height(height), m_hdr(hdr)
	{
		glGenFramebuffers(1, &m_id);

		m_colorTexIndex.resize(1);
		setupColorFramebuffer("color", 0);

		glGenRenderbuffers(1, &m_rbo);
		glBindRenderbuffer(GL_RENDERBUFFER, m_rbo);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, m_width, m_height);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);
		glBindFramebuffer(GL_FRAMEBUFFER, m_id);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_rbo);
		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			PRINT_ERROR("Framebuffer not complete!");
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void FrameBuffer::bind()
	{
		glViewport(0, 0, m_width, m_height);
		glBindFramebuffer(GL_FRAMEBUFFER, m_id);
	}

	void FrameBuffer::unBind(int width, int height)
	{
		if (width == -1)width = m_width;
		if (height == -1)height = m_height;
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glViewport(0, 0, width, height);
	}


	void FrameBuffer::setupDepthFramebuffer(const std::string& name)
	{
		TextureManager::ptr textureMgr = TextureManager::getSingleton();
		m_depthTexIndex = textureMgr->loadTextureDepth(name, m_width, m_height);
		glBindFramebuffer(GL_FRAMEBUFFER, m_id);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
			GL_TEXTURE_2D, textureMgr->getTexture(m_depthTexIndex)->getTextureId(), 0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
	
	void FrameBuffer::setupStencilFramebuffer(const std::string& name)
	{
		TextureManager::ptr textureMgr = TextureManager::getSingleton();
		m_stencilTexIndex = textureMgr->loadTextureStencil(name, m_width, m_height);
		glBindFramebuffer(GL_FRAMEBUFFER, m_id);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT,
			GL_TEXTURE_2D, textureMgr->getTexture(m_stencilTexIndex)->getTextureId(), 0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	
	}

	void FrameBuffer::setupColorFramebuffer(const std::string& name, unsigned int attachIdx)
	{
		TextureManager::ptr textureMgr = TextureManager::getSingleton();
		m_colorTexIndex[attachIdx] = textureMgr->loadTextureColor(name, m_width, m_height, m_hdr);
		glBindFramebuffer(GL_FRAMEBUFFER, m_id);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + attachIdx,
			GL_TEXTURE_2D, textureMgr->getTexture(m_colorTexIndex[attachIdx])->getTextureId(), 0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void FrameBuffer::clearFramebuffer()
	{
		glDeleteFramebuffers(1, &m_id);
	}

	void FrameBuffer::saveTextureToFile(const std::string& filename, TextureType type)
	{
		// 绑定帧缓冲
		glBindFramebuffer(GL_FRAMEBUFFER, m_id);
		GLenum attachment = (type == TextureType::COLOR) ? GL_COLOR_ATTACHMENT0 : GL_DEPTH_ATTACHMENT;
		GLuint textureIndex = (type == TextureType::COLOR) ? m_colorTexIndex[0] : m_depthTexIndex;
		glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, 
			TextureManager::getInstance()->getTexture(m_depthTexIndex)->getTextureId(), 0);

		// 读取像素数据
		std::vector<float> pixels(m_width * m_height * (type == TextureType::COLOR ? 4 : 1)); // RGBA或单通道深度
		GLenum format = (type == TextureType::COLOR) ? GL_RGBA : GL_DEPTH_COMPONENT;
		GLenum dataType = GL_FLOAT;
		glReadPixels(0, 0, m_width, m_height, format, dataType, pixels.data());

		// 翻转像素数据
		int rowSize = m_width * (type == TextureType::COLOR ? 4 : 1);
		std::vector<float> flippedPixels(pixels.size());
		for (int y = 0; y < m_height; y++) {
			std::memcpy(&flippedPixels[y * rowSize], &pixels[(m_height - 1 - y) * rowSize], rowSize * sizeof(float));
		}

		// 保存为图像文件
		if (type == TextureType::COLOR) {
			stbi_write_png(filename.c_str(), m_width, m_height, 4, flippedPixels.data(), m_width * 4);
		}
		else {
			// 对于深度纹理，可能需要转换数据格式或进行缩放
			// 这里只是一个简单的示例，可能需要根据实际情况调整
			std::vector<unsigned char> depthPixels(m_width * m_height);
			for (size_t i = 0; i < depthPixels.size(); ++i) {
				depthPixels[i] = static_cast<unsigned char>(255.0f * flippedPixels[i]);
			}
			stbi_write_png(filename.c_str(), m_width, m_height, 1, depthPixels.data(), m_width);
		}

		// 解绑帧缓冲
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
}