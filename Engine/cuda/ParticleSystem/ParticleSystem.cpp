#ifdef _WIN32
#include "glad/glad.h"
#elif defined(__linux__)
#include <GL/glew.h>
#endif
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "core/base.h"
#include "ParticleSystem/ParticleSystem.h"

Physics::ParticleSystem::~ParticleSystem()
{
	checkCudaErrors(cudaGraphicsUnregisterResource(m_cudaVBOResource));
	glDeleteBuffers(1, &m_particleVBO);
}
