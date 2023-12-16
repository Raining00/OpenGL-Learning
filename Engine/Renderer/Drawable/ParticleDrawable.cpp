#include "Drawable/ParticleDrawable.h"
#include "include/Config.h"
#include "include/ColorfulPrint.h"

namespace Renderer
{
	ParticlePointSpriteDrawable::ParticlePointSpriteDrawable(unsigned int posChannel):m_posChannel(posChannel)
	{
		m_particleVBO = 0;
		m_numParticles = 0;
		m_particleRadius = 1.0f;
		m_vboCreateBySelf = false;
		m_baseColor = glm::vec3(1.0f, 0.6f, 0.3f);
		glGenVertexArrays(1, &m_particleVAO);

		// load shader and texture
		m_shaderManager = ShaderManager::getInstance();
		m_textureManager = TextureManager::getInstance();
		m_shaderManager->loadShader("Particle", SHADER_PATH"/Particles/ParticlePointSprit.vs", SHADER_PATH"/Particles/ParticlePointSprit.fs");
		generateGaussianMap(32);
	}

	ParticlePointSpriteDrawable::~ParticlePointSpriteDrawable()
	{
		if (m_vboCreateBySelf && m_particleVBO != 0)
			glDeleteBuffers(1, &m_particleVBO);
		glDeleteVertexArrays(1, &m_particleVAO);
	}

	void ParticlePointSpriteDrawable::generateGaussianMap(int resolution)
	{
		// generate gaussian map manually.
		unsigned char* data = new unsigned char[4 * resolution * resolution];
		int index = 0;
		float step = 2.0f / resolution;
		float X, Y, value;
		float distance;
		Y = -1.0f;
		for (int y = 0; y < resolution; ++y, Y += step)
		{
			float Y2 = Y * Y;
			X = -1.0f;
			for (int x = 0; x < resolution; ++x, X += step)
			{
				distance = sqrtf(X * X + Y2);
				if (distance > 1.0f)distance = 1.0f;
				value = 2 * pow(distance, 3.0f) - 3 * pow(distance, 2.0f) + 1.0f;
				data[index + 0] = static_cast<char>(value * 255);
				data[index + 1] = static_cast<char>(value * 255);
				data[index + 2] = static_cast<char>(value * 255);
				data[index + 3] = static_cast<char>(value * 255);
				index += 4;
			}
		}

		m_particleTexture = m_textureManager->loadTexture2D("GaussianMap", data, resolution, resolution, 4);
		delete[]data;
	}


}