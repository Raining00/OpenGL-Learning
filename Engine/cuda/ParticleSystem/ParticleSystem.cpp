#ifdef _WIN32
#include "glad/glad.h"
#elif defined(__linux__)
#include <GL/glew.h>
#endif
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "core/base.h"
#include "include/ColorfulPrint.h"
#include "ParticleSystem/ParticleSystem.h"

Physics::ParticleSystem::~ParticleSystem()
{
	if (!m_isInit) return;
	checkCudaErrors(cudaGraphicsUnregisterResource(m_cudaVBOResource));
	glDeleteBuffers(1, &m_PosVBO);
}

bool Physics::ParticleSystem::initialize(unsigned int particleNum)
{
	checkCudaErrors(cudaMalloc((void**)&m_devPosition, sizeof(float) * 4 * particleNum));
	checkCudaErrors(cudaMemset(m_devPosition, 0, particleNum));
	checkCudaErrors(cudaMalloc((void**)&m_devVelocity, sizeof(float) * 4));
	checkCudaErrors(cudaMemset(m_devVelocity, 0, particleNum));
	m_particleNum = particleNum;
	glGenBuffers(1, &m_PosVBO);
	glBindBuffer(GL_ARRAY_BUFFER, m_PosVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * particleNum, nullptr, GL_DYNAMIC_DRAW);
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_cudaVBOResource, m_PosVBO, cudaGraphicsMapFlagsWriteDiscard));
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	m_isInit = true;
	return true;
}

bool Physics::ParticleSystem::reset()
{
	if (!m_isInit) return false;

	checkCudaErrors(cudaGraphicsUnregisterResource(m_cudaVBOResource));
	glBindBuffer(GL_ARRAY_BUFFER, m_PosVBO);
	glBufferSubData(GL_ARRAY_BUFFER, 0, m_hostPosition.size() * sizeof(float) * 4, m_hostPosition.data());
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_cudaVBOResource, m_PosVBO, cudaGraphicsMapFlagsNone));
	checkCudaErrors(cudaMemcpy(m_devVelocity, m_hostVelocity.data(), sizeof(float) * 4 * m_hostVelocity.size(), cudaMemcpyHostToDevice));

	return true;
}

void Physics::ParticleSystem::setParticlePosition(const std::vector<glm::vec4>& hostPosition)
{
	if (hostPosition.size() != m_particleNum)
	{
		if (m_isInit)
		{
			checkCudaErrors(cudaFree(m_devPosition));
			checkCudaErrors(cudaFree(m_devVelocity));
			checkCudaErrors(cudaGraphicsUnregisterResource(m_cudaVBOResource));
			glDeleteBuffers(1, &m_PosVBO);
		}
		m_particleNum = hostPosition.size();
		checkCudaErrors(cudaMalloc((void**)&m_devPosition, sizeof(float) * 4 * m_particleNum));
		checkCudaErrors(cudaMalloc((void**)&m_devVelocity, sizeof(float) * 4 * m_particleNum));
		glGenBuffers(1, &m_PosVBO);
		glBindBuffer(GL_ARRAY_BUFFER, m_PosVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * m_particleNum, nullptr, GL_DYNAMIC_DRAW);
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_cudaVBOResource, m_PosVBO, cudaGraphicsMapFlagsWriteDiscard));
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	checkCudaErrors(cudaGraphicsUnregisterResource(m_cudaVBOResource));
	glBindBuffer(GL_ARRAY_BUFFER, m_PosVBO);
	glBufferSubData(GL_ARRAY_BUFFER, 0, hostPosition.size() * sizeof(float) * 4, hostPosition.data());
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_cudaVBOResource, m_PosVBO, cudaGraphicsMapFlagsNone));
	m_hostPosition = hostPosition;
}

void Physics::ParticleSystem::setParticlePosition(float* hostPosition, unsigned int particleNum)
{
	if (particleNum != m_particleNum)
	{
		if (m_isInit)
		{
			checkCudaErrors(cudaFree(m_devPosition));
			checkCudaErrors(cudaFree(m_devVelocity));
			checkCudaErrors(cudaGraphicsUnregisterResource(m_cudaVBOResource));
			glDeleteBuffers(1, &m_PosVBO);
		}
		m_particleNum = particleNum;
		checkCudaErrors(cudaMalloc((void**)&m_devPosition, sizeof(float) * 4 * m_particleNum));
		checkCudaErrors(cudaMalloc((void**)&m_devVelocity, sizeof(float) * 4 * m_particleNum));
		glGenBuffers(1, &m_PosVBO);
		glBindBuffer(GL_ARRAY_BUFFER, m_PosVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * m_particleNum, nullptr, GL_DYNAMIC_DRAW);
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_cudaVBOResource, m_PosVBO, cudaGraphicsMapFlagsWriteDiscard));
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	checkCudaErrors(cudaGraphicsUnregisterResource(m_cudaVBOResource));
	glBindBuffer(GL_ARRAY_BUFFER, m_PosVBO);
	glBufferSubData(GL_ARRAY_BUFFER, 0, particleNum * sizeof(float) * 4, hostPosition);
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_cudaVBOResource, m_PosVBO, cudaGraphicsMapFlagsNone));
	//m_hostPosition.assign(hostPosition, hostPosition + particleNum);
}

void Physics::ParticleSystem::setParticleVelocity(const std::vector<glm::vec4>& hostVelocity)
{
	if (m_particleNum != hostVelocity.size())
	{
		PRINT_ERROR("SIZE MISMATCH! The number of particles is " << m_particleNum << " While size of the velocity array is " << hostVelocity.size());
		PRINT_ERROR("Make sure the velocity array have the same size of the position array. Velocity setting failed!");
		return;
	}
	checkCudaErrors(cudaMemcpy(m_devPosition, hostVelocity.data(), sizeof(float) * 4 * hostVelocity.size(), cudaMemcpyHostToDevice));
	m_hostVelocity = hostVelocity;
}

void Physics::ParticleSystem::setParticleVelocity(float* hostVelocity, unsigned int particleNum)
{
	if (m_particleNum != particleNum)
	{
		PRINT_ERROR("SIZE MISMATCH! The number of particles is " << m_particleNum << " While size of the velocity array is " << particleNum);
		PRINT_ERROR("Make sure the velocity array have the same size of the position array. Velocity setting failed!");
		return;
	}
	checkCudaErrors(cudaMemcpy(m_devPosition, hostVelocity, sizeof(float) * 4 * particleNum, cudaMemcpyHostToDevice));
	//m_hostVelocity.assign(hostVelocity, hostVelocity + particleNum);
}



