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
		ParticlePointSpriteDrawable(unsigned int posChannel = 4);

		~ParticlePointSpriteDrawable();

		void setParticleRadius(float radius);
		void setPositions(const std::vector<glm::vec3>& positions);
		void setPositions(const std::vector<glm::vec4>& positions);
		void setPositions(CArray<Vec3f> &positions);
		void setPositions(std::vector<Vec3f> &positions);

		void setPositionsFromDecvice(const float* positions, unsigned int numParticles, unsigned int posChannel);
		void setPositionsFromDevice(DArray<Vec3f> &positions);

		void setParticleVBO(GLuint vbo);

		glm::vec3 &getBaseColor() { return m_baseColor; }
		GLuint getParticleVBO() { return m_particleVBO; }
		void setBaseColor(const glm::vec3 &color) { m_baseColor = color; }

		virtual void render(Camera3D::ptr camera, Camera3D::ptr lightCamera, Shader::ptr shader = nullptr);
		virtual void renderDepth(Shader::ptr shader, Camera3D::ptr lightCamera);

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
		bool m_vboCreateBySelf;
		unsigned int m_numParticles;
		unsigned int m_posChannel; // 3 or 4 (vec3 or vec4)
	};
}