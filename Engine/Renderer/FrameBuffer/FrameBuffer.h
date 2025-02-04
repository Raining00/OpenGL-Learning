#pragma once

#include <vector>
#ifdef _WIN32
#include "glad/glad.h"
#elif defined(__linux__)
#include "GL/glew.h"
#endif 
#include <memory>
#include <string>

namespace Renderer
{
	enum class TextureType;

	const GLenum ColorAttachments[] =
	{
		GL_COLOR_ATTACHMENT0,
		GL_COLOR_ATTACHMENT1,
		GL_COLOR_ATTACHMENT2,
		GL_COLOR_ATTACHMENT3,
		GL_COLOR_ATTACHMENT4,
		GL_COLOR_ATTACHMENT5,
		GL_COLOR_ATTACHMENT6,
		GL_COLOR_ATTACHMENT7
	};

	class FrameBuffer
	{
	public:
		typedef std::shared_ptr<FrameBuffer> ptr;
		FrameBuffer(int width, int height, const std::string& depthName, const std::string& stencilName, const std::vector<std::string>& colorNames, bool hdr = false);
		FrameBuffer(int width, int height, TextureType type, bool hdr);
		FrameBuffer(int width, int height, bool hdr = false);
		virtual ~FrameBuffer() { clearFramebuffer(); }

		void bind();
		void unBind(int width = -1, int height = -1);

		unsigned int getFramebufferId() const { return m_id; }
		unsigned int getDepthTextureIndex() const { return m_depthTexIndex; }
		unsigned int getColorTextureIndex(int idx) const { return m_colorTexIndex[idx]; }
		unsigned int getColorTextureBuffer() const { return m_textureColorbuffer; }
		void saveTextureToFile(const std::string& filename, TextureType type);
		void saveDepthCubeTexture(const std::string& filename);

	private:
		void setupDepthFramebuffer(const std::string& depthName);
		void setupStencilFramebuffer(const std::string& stencilName);
		void setupColorFramebuffer(const std::string& name, unsigned int attachIdx);
		void setupDepthCubeFrameBuffer(const std::string& depthCubeName);
		void clearFramebuffer();

	private:
		bool m_hdr;
		unsigned int m_id, m_rbo, m_textureColorbuffer;
		int m_width, m_height;
		unsigned int m_depthTexIndex, m_stencilTexIndex, m_depthCubeTexIndex;
		std::vector<unsigned int> m_colorTexIndex;
	};
}