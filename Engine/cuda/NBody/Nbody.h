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
    void init(int numBodies);
    void reset(int numBodies);
private:
    T* m_hPos;
    T* m_hVel;
    unsigned int m_numBodies;
    std::shared_ptr<BodySystemCUDA<T>> m_nbodyCUDA;
};

template <typename T>
NBodySystem<T>::NBodySystem()
{
    cudaDeviceProp props;
    int devID;
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));

}

template<typename T>
inline void NBodySystem<T>::init(int numBodies)
{
}
