#pragma once
#include "bodysystemcuda.h"
#include "../core/base.h"
#include <memory>
NBodyParams demoParams[] = {
    {0.016f, 1.54f, 8.0f, 0.1f, 1.0f, 1.0f, 0, -2, -100},
    {0.016f, 0.68f, 20.0f, 0.1f, 1.0f, 0.8f, 0, -2, -30},
    {0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    {0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    {0.0019f, 0.32f, 276.0f, 1.0f, 1.0f, 0.07f, 0, 0, -5},
    {0.0016f, 0.32f, 272.0f, 0.145f, 1.0f, 0.08f, 0, 0, -5},
    {0.016000f, 6.040000f, 0.000000f, 1.000000f, 1.000000f, 0.760000f, 0, 0,
     -50},
};

template <typename T>
class NBodySystem
{
public:
    NBodySystem();
    ~NBodySystem();
    void reset(int numBodies);
    // get methods
    T* getHostPositions() const { return m_hPos; }
    T* getHostVelocities() const { return m_hVel; }
    float* getHostColors() const { return m_hColor; }
    unsigned int getNumBodies() const { return m_numBodies; }

    // set methods
    void setActiveParams(int activeParams) { if(activeParams<7) m_activeParams = demoParams[activeParams]; }
protected:
    void _init();

private:
    T* m_hPos{ nullptr };
    T* m_hVel{ nullptr };
    float* m_hColor{ nullptr };
    unsigned int m_numBodies;
    int m_blockSize{ 0 };
    int m_numIterations{ 0 };
    int m_activateDemo{ 0 };
    bool bSupportDouble{ true };
    NBodyParams m_activeParams;
    std::shared_ptr<BodySystemCUDA<T>> m_nbodyCUDA;
};

template <typename T>
NBodySystem<T>::NBodySystem()
{
    m_activeParams = demoParams[m_activateDemo];

    cudaDeviceProp props;
    int devID;
    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    // CC 1.2 and earlier do not support double precision
    if (props.major * 10 + props.minor <= 12) {
        bSupportDouble = false;
    }
    if (m_blockSize == 0)  // blockSize not set on command line
        m_blockSize = 256;
    m_numBodies = m_blockSize * 4 * props.multiProcessorCount;

    if (m_numBodies <= 1024) {
        m_activeParams.m_clusterScale = 1.52f;
        m_activeParams.m_velocityScale = 2.f;
    }
    else if (m_numBodies <= 2048) {
        m_activeParams.m_clusterScale = 1.56f;
        m_activeParams.m_velocityScale = 2.64f;
    }
    else if (m_numBodies <= 4096) {
        m_activeParams.m_clusterScale = 1.68f;
        m_activeParams.m_velocityScale = 2.98f;
    }
    else if (m_numBodies <= 8192) {
        m_activeParams.m_clusterScale = 1.98f;
        m_activeParams.m_velocityScale = 2.9f;
    }
    else if (m_numBodies <= 16384) {
        m_activeParams.m_clusterScale = 1.54f;
        m_activeParams.m_velocityScale = 8.f;
    }
    else if (m_numBodies <= 32768) {
        m_activeParams.m_clusterScale = 1.44f;
        m_activeParams.m_velocityScale = 11.f;
    }
    this->_init();
}

template <typename T>
void NBodySystem<T>::_init()
{
    //m_nbodyCUDA = std::make_shared<BodySystemCUDA<T>>(m_numBodies, m_blockSize);
    
    if(m_hPos != nullptr)
		delete[] m_hPos;
    if(m_hVel != nullptr)
        delete[] m_hVel;
    if(m_hColor != nullptr)
		delete[] m_hColor;

    // allocate host memory
    m_hPos = new T[m_numBodies * 4];
    m_hVel = new T[m_numBodies * 4];
    m_hColor = new float[m_numBodies * 4];

    randomizeBodies(NBODY_CONFIG_SHELL, m_hPos, m_hVel, m_hColor, m_activeParams.m_clusterScale, m_activeParams.m_velocityScale, m_numBodies, true);
}

template <typename T>
void NBodySystem<T>::reset(int numBodies)
{
	m_numBodies = numBodies;
	this->_init();
}

template <typename T>
NBodySystem<T>::~NBodySystem()
{
	if (m_hPos)
		delete[] m_hPos;
	if (m_hVel)
		delete[] m_hVel;
	if (m_hColor)
		delete[] m_hColor;
}