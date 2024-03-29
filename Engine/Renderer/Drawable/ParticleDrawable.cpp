#include "Drawable/ParticleDrawable.h"
#include "RenderApp/RenderDevice.h"

#include "include/Config.h"
#include "include/ColorfulPrint.h"

#include "Engine/cuda/core/base.h"

namespace Renderer
{
	ParticlePointSpriteDrawable::ParticlePointSpriteDrawable(const unsigned int& posChannel, const bool& Glow):
		m_posChannel(posChannel), m_glow(Glow)
	{
		m_baseColor = glm::vec3(1.0f, 0.6f, 0.3f);
		glGenVertexArrays(1, &m_particleVAO);
		m_vboCreateBySelf = false;

		// load shader and texture
		m_shaderManager = ShaderManager::getInstance();
		m_textureManager = TextureManager::getInstance();
		std::string vertexPath;
		if(m_posChannel == 3)
			vertexPath = SHADER_PATH "/Particles/ParticlePointSpriteVec3.vs";
		else if(m_posChannel == 4)
			vertexPath = SHADER_PATH "/Particles/ParticlePointSpriteVec4.vs";
		else
			throw std::runtime_error("ParticlePointSpriteDrawable::ParticlePointSpriteDrawable: posChannel must be 3 or 4.");
		m_shaderIndex = m_shaderManager->loadShader("ParticlePointSprite", vertexPath.c_str(), SHADER_PATH "/Particles/ParticlePointSprite.fs");
		m_shaderManager->loadShader("ParticlePix", vertexPath.c_str(), SHADER_PATH"/Particles/ParticlePix.fs");
		if(Glow)
			m_shaderIndex = m_shaderManager->getShdaerIndex("ParticlePix");
		generateGaussianMap(32);
	}

	ParticlePointSpriteDrawable::~ParticlePointSpriteDrawable()
	{
		if (m_vboCreateBySelf && m_particleVBO != 0)
			glDeleteBuffers(1, &m_particleVBO);
		glDeleteVertexArrays(1, &m_particleVAO);
	}

	void ParticlePointSpriteDrawable::Initialize(const unsigned int& particleNum, const float& particleRadius, const unsigned int& posChannel)
	{
		m_numParticles = particleNum;
		m_posChannel = posChannel;
		m_particleRadius = particleRadius;
		m_baseColor = glm::vec3(1.0f, 0.6f, 0.3f);
		glGenBuffers(1, &m_particleVBO);
		glGenBuffers(1, &m_ColorVBO);
		m_vboCreateBySelf = true;

		glBindVertexArray(m_particleVAO);
		glBindBuffer(GL_ARRAY_BUFFER, m_particleVBO);
		glBufferData(GL_ARRAY_BUFFER, m_numParticles * m_posChannel * sizeof(float), nullptr, GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, m_posChannel, GL_FLOAT, GL_FALSE, m_posChannel * sizeof(float), (void*)0);
		glVertexAttribDivisor(0, 1);

		// color attribute
		float* colors = new float[m_numParticles * 4];
		for (int i = 0; i < m_numParticles * 4; i += 4) {
			colors[i] = 1.0f;     // R
			colors[i + 1] = 1.0f; // G
			colors[i + 2] = 1.0f; // B
			colors[i + 3] = 1.0f; // A
		}
		glBindBuffer(GL_VERTEX_ARRAY, m_ColorVBO);
		glBufferData(GL_ARRAY_BUFFER, m_numParticles * 4 * sizeof(float), colors, GL_STATIC_DRAW);
		glEnableVertexAttribArray(3);
		glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
		glVertexAttribDivisor(3, 1);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);

		// load shader and texture
		m_shaderManager = ShaderManager::getInstance();
		m_textureManager = TextureManager::getInstance();
		std::string vertexPath;
		if (m_posChannel == 3)
			vertexPath = SHADER_PATH "/Particles/ParticlePointSpriteVec3.vs";
		else if (m_posChannel == 4)
			vertexPath = SHADER_PATH "/Particles/ParticlePointSpriteVec4.vs";
		else
			throw std::runtime_error("ParticlePointSpriteDrawable::ParticlePointSpriteDrawable: posChannel must be 3 or 4.");
		m_shaderIndex = m_shaderManager->loadShader("ParticlePointSprite", vertexPath.c_str(), SHADER_PATH "/Particles/ParticlePointSprite.fs");

	}

	void ParticlePointSpriteDrawable::setParticlePositions(const std::vector<glm::vec3>& positions)
	{
		if(m_posChannel != 3)
			throw std::runtime_error("ParticlePointSpriteDrawable::setParticlePositions: posChannel must be 3.");
		glBindBuffer(GL_ARRAY_BUFFER, m_particleVBO);
		glBufferSubData(GL_ARRAY_BUFFER, 0, positions.size() * sizeof(glm::vec3), positions.data());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	void ParticlePointSpriteDrawable::setParticlePositions(const std::vector<glm::vec4>& positions)
	{
		if (m_posChannel != 4)
			throw std::runtime_error("ParticlePointSpriteDrawable::setParticlePositions: posChannel must be 4.");
		glBindBuffer(GL_ARRAY_BUFFER, m_particleVBO);
		glBufferSubData(GL_ARRAY_BUFFER, 0, positions.size() * sizeof(glm::vec4), positions.data());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	void ParticlePointSpriteDrawable::setParticlePositions(CArray<Vec3f>& positions)
	{
		if (m_posChannel != 3)
			throw std::runtime_error("ParticlePointSpriteDrawable::setParticlePositions: posChannel must be 3.");
		glBindBuffer(GL_ARRAY_BUFFER, m_particleVBO);
		glBufferSubData(GL_ARRAY_BUFFER, 0, positions.size() * sizeof(Vec3f), positions.begin());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	void ParticlePointSpriteDrawable::setParticlePositions(CArray<Vec4f>& positions)
	{
		if (m_posChannel != 4)
			throw std::runtime_error("ParticlePointSpriteDrawable::setParticlePositions: posChannel must be 4.");
		glBindBuffer(GL_ARRAY_BUFFER, m_particleVBO);
		glBufferSubData(GL_ARRAY_BUFFER, 0, positions.size() * sizeof(Vec4f), positions.begin());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	void ParticlePointSpriteDrawable::setParticlePositions(std::vector<Vec3f>& positions)
	{
		if (m_posChannel != 3)
			throw std::runtime_error("ParticlePointSpriteDrawable::setParticlePositions: posChannel must be 3.");
		glBindBuffer(GL_ARRAY_BUFFER, m_particleVBO);
		glBufferSubData(GL_ARRAY_BUFFER, 0, positions.size() * sizeof(Vec3f), positions.data());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	void ParticlePointSpriteDrawable::setParticlePositions(std::vector<Vec4f>& positions)
	{
		if (m_posChannel != 4)
			throw std::runtime_error("ParticlePointSpriteDrawable::setParticlePositions: posChannel must be 4.");
		glBindBuffer(GL_ARRAY_BUFFER, m_particleVBO);
		glBufferSubData(GL_ARRAY_BUFFER, 0, positions.size() * sizeof(Vec4f), positions.data());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	void ParticlePointSpriteDrawable::setGlow(const bool& glow)
	{
		if (glow)
			m_shaderIndex = m_shaderManager->getShdaerIndex("ParticlePointSprite");
		else
			m_shaderIndex = m_shaderManager->getShdaerIndex("ParticlePix");
	}

	void ParticlePointSpriteDrawable::setColor(float* color, int numParticles)
	{
		// check array size
		if (numParticles != m_numParticles)
		{
			std::cout << "ParticlePointSpriteDrawable::setColor: numParticles != m_numParticles." << std::endl;
			return;
		}
		glBindBuffer(GL_ARRAY_BUFFER, m_ColorVBO);
		glBufferSubData(GL_ARRAY_BUFFER, 0, numParticles * 4 * sizeof(float), color);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	void ParticlePointSpriteDrawable::setColor(std::vector<glm::vec3>& color)
	{
		// check array size
		if (color.size() != m_numParticles)
		{
			std::cout << "ParticlePointSpriteDrawable::setColor: color.size() != m_numParticles." << std::endl;
			return;
		}
		glBindBuffer(GL_ARRAY_BUFFER, m_ColorVBO);
		glBufferSubData(GL_ARRAY_BUFFER, 0, color.size() * 4 * sizeof(float), color.data());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	void ParticlePointSpriteDrawable::setParticleVBO(GLuint vbo, int numParticles)
	{
		if (m_vboCreateBySelf && m_particleVBO != 0)
		{
			glDeleteBuffers(1, &m_particleVBO);
		}
		m_particleVBO = vbo;
		m_vboCreateBySelf = false;
		m_numParticles = numParticles;
		m_posChannel = 4;
		glBindVertexArray(m_particleVAO);
		glBindBuffer(GL_ARRAY_BUFFER, m_particleVBO);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, m_posChannel, GL_FLOAT, GL_FALSE, m_posChannel * sizeof(float), (void*)0);
		glVertexAttribDivisor(0, 1);
		glBindVertexArray(0);
	}

	void ParticlePointSpriteDrawable::render(Camera3D::ptr camera, Camera3D::ptr lightCamera, Shader::ptr shader)
	{
		if (!m_visible) return;

		// calculate particle size scale factor
		float aspect = camera->getAspect();
		float fovy = camera->getFovy();
		int width = RenderDevice::getInstance()->getWindowWidth();
		float pointScale = 1.0f * width / aspect * (1.0f / tanf(glm::radians(fovy) * 0.5f));
		glEnable(GL_PROGRAM_POINT_SIZE);
		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
		glDisable(GL_CULL_FACE);
		// disable stencil test.
		glStencilMask(0x00);
		if (m_stencil)
		{
			glEnable(GL_STENCIL_TEST);
			glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
			glStencilFunc(GL_ALWAYS, 1, 0xFF);
			glStencilMask(0xFF);
		}
		shader = m_shaderManager->getShader(m_shaderIndex);
		shader->use();
		LightManager::getInstance()->setLight(shader, camera);
		shader->setInt("image", 0);
		shader->setFloat("pointScale", pointScale);
		shader->setVec3("baseColor", m_baseColor);
		shader->setFloat("pointSize", m_particleRadius);
		shader->setBool("glow", m_glow);
		Texture::ptr depthMap = m_textureManager->getTexture("shadowDepth");
		if (depthMap != nullptr)
		{
			shader->setInt("shadowMap", 5);
			depthMap->bind(5);
		}
		// light space matrix.
		if (lightCamera != nullptr)
			shader->setMat4("lightSpaceMatrix",
				lightCamera->getProjectionMatrix() * lightCamera->getViewMatrix());
		else
			shader->setMat4("lightSpaceMatrix", glm::mat4(1.0f));
		// object matrix.
		shader->setBool("instance", m_instance);
		shader->setBool("receiveShadow", m_receiveShadow);
		shader->setMat4("modelMatrix", m_transformation.getWorldMatrix());
		shader->setMat3("normalMatrix", m_transformation.getNormalMatrix());

		// bind particle texture.
		m_textureManager->bindTexture(m_particleTexture, 0);

		// draw
		glBindVertexArray(m_particleVAO);
		glDrawArraysInstanced(GL_POINTS, 0, 1, m_numParticles);
		glBindVertexArray(0);

		// restore
		m_shaderManager->unbindShader();
		m_textureManager->unbindTexture(m_particleTexture);
		glDisable(GL_PROGRAM_POINT_SIZE);
		glEnable(GL_CULL_FACE);
	}

	void ParticlePointSpriteDrawable::renderDepth(Shader::ptr shader, Camera3D::ptr lightCamera)
	{
		if (!m_visible || !m_produceShadow)
			return;
		std::string vertexPath;
		if(m_posChannel == 3)
			vertexPath = SHADER_PATH "/Particles/spriteDepthVec3.vs";
		else if(m_posChannel == 4)
			vertexPath = SHADER_PATH "/Particles/spriteDepthVec4.vs";
		else
			throw std::runtime_error("ParticlePointSpriteDrawable::renderDepth: posChannel must be 3 or 4.");
		static unsigned int index = m_shaderManager->loadShader("spriteDepth", vertexPath.c_str(),
			SHADER_PATH"/Particles/spriteDepth.fs");
		shader = m_shaderManager->getShader(index);
		shader->use();
		shader->setInt("particleTexture", 0);
		shader->setFloat("pointSize", m_particleRadius);
		shader->setMat4("lightSpaceMatrix",
			lightCamera->getProjectionMatrix() * lightCamera->getViewMatrix());
		shader->setMat4("modelMatrix", m_transformation.getWorldMatrix());
		m_textureManager->bindTexture(m_particleTexture, 0);
		glDisable(GL_CULL_FACE);
		glEnable(GL_PROGRAM_POINT_SIZE);
		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
		glBindVertexArray(m_particleVAO);
		glDrawArraysInstanced(GL_POINTS, 0, 1, m_numParticles);
		glBindVertexArray(0);
		glDisable(GL_PROGRAM_POINT_SIZE);
		glEnable(GL_CULL_FACE);
		m_shaderManager->unbindShader();
		m_textureManager->unbindTexture(m_particleTexture);
	}

	void ParticlePointSpriteDrawable::renderDepthCube(Shader::ptr shader)
	{
		if (!m_visible || !m_produceShadow)
			return;
		shader->use();
		shader->setBool("instance", m_instance);
		shader->setMat4("modelMatrix", m_transformation.getWorldMatrix());
		this->renderImp();
		ShaderManager::getSingleton()->unbindShader();
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