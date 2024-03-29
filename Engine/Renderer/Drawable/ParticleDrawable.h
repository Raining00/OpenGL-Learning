#pragma once

#include "Drawable/Drawable.h"
#include "Manager/ShaderManager.h"
#include "Manager//TextureManager.h"

#include "Engine/cuda/core/Array.h"
#include "Engine/cuda/core/Vector/Vector.h"

namespace Renderer
{
	class ParticlePointSpriteDrawable : public Drawable
	{
	public:
		ParticlePointSpriteDrawable() = default;
		ParticlePointSpriteDrawable(const unsigned int& posChannel = 4, const bool& Glow = false);

		~ParticlePointSpriteDrawable();

		void setParticleRadius(const float& radius) { m_particleRadius = radius; }
		void Initialize(const unsigned int& particleNum, const float& particleRadius = 1, const unsigned int& posChannel = 4);
		void setParticlePositions(const std::vector<glm::vec3>& positions);
		void setParticlePositions(const std::vector<glm::vec4>& positions);

		void setParticlePositions(CArray<Vec3f> &positions);
		void setParticlePositions(CArray<Vec4f> &positions);

		void setParticlePositions(std::vector<Vec3f> &positions);
		void setParticlePositions(std::vector<Vec4f> &positions);
		void setGlow(const bool& glow);
		void setColor(float* color, int numParticles);
		void setColor(std::vector<glm::vec3> &color);

		void setParticleVBO(GLuint vbo, int numParticles);

		glm::vec3 &getBaseColor() { return m_baseColor; }
		GLuint getParticleVBO() { return m_particleVBO; }
		void setBaseColor(const glm::vec3 &color) { m_baseColor = color; }

		virtual void render(Camera3D::ptr camera, Camera3D::ptr lightCamera, Shader::ptr shader = nullptr) override;
		virtual void renderDepth(Shader::ptr shader, Camera3D::ptr lightCamera) override;
		virtual void renderDepthCube(Shader::ptr shader) override;

	private:
		void generateGaussianMap(int resolution);
	private:
		ShaderManager::ptr m_shaderManager;
		TextureManager::ptr m_textureManager;
		glm::vec3 m_baseColor;
		float m_particleRadius;
		GLuint m_particleTexture;
		GLuint m_particleVAO;
		GLuint m_particleVBO;
		GLuint m_ColorVBO;
		bool m_vboCreateBySelf, m_glow{ false };
		unsigned int m_numParticles;
		unsigned int m_posChannel; // 3 or 4 (vec3 or vec4)
	};
}