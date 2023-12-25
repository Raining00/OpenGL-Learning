#pragma once

#include "FrameBuffer/FrameBuffer.h"

namespace Renderer
{
	class GaussianBlur
	{
	public:
		typedef std::shared_ptr<GaussianBlur> ptr;

		GaussianBlur(int width, int height);
		~GaussianBlur() = default;

		void renderGaussianBlurEffect(const GLuint& blurTexIndex);

		unsigned int getFrameBufferId()const { return m_pingPongBuffer[0]->getFramebufferId(); }
		unsigned int getSceneTexIndex()const;
		unsigned int& getBlurTimes() { return m_blurTimes; }
		void setBlurTimes(const unsigned int& t);

	private:
		unsigned int m_blurTimes;
		unsigned int m_screenQuadIndex;
		unsigned int m_gaussianShaderIndex;
		unsigned int m_readBuffer, m_writeBuffer;
		int m_width, m_height;
		FrameBuffer::ptr m_pingPongBuffer[2];
	};
}