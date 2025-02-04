#pragma once

#include "bodysystem.h"

template <typename T>
struct DeviceData
{
	T* dPos[2];
	T* dVel;
	cudaEvent_t event;
	unsigned int offset;
	unsigned int numBodies;
};

template <typename T>
class BodySystemCUDA :public BodySystem<T>
{
public:
	BodySystemCUDA(unsigned int numBodies, unsigned int blockSize, bool useSysMem = false, int deviceId = 0);
	virtual ~BodySystemCUDA();

	virtual void update(T delraTime);

	virtual void setSoftening(T softening);
	virtual void setDamping(T damping);

	virtual T* getArray(BodyArray array);
	virtual void setArray(BodyArray array, const T* deta);

	virtual unsigned int getCurrentReadBuffer() const {
		return m_pbo[m_currentRead];
	}

	virtual unsigned int getNumBodies() const { return m_numBodies; }

protected:  // methods
	BodySystemCUDA() {}

	virtual void _initialize(int numBodies);
	virtual void _finalize();

protected:  // data
	unsigned int m_numBodies;
	unsigned int m_numDevices;
	bool m_bInitialized;
	int m_devID;

	// Host data
	T* m_hPos[2];
	T* m_hVel;

	DeviceData<T>* m_deviceData;

	bool m_bUsePBO;
	bool m_bUseSysMem;
	bool m_bUseP2P;
	unsigned int m_SMVersion;

	T m_damping;

	unsigned int m_pbo[2];
	cudaGraphicsResource* m_pGRes[2];
	unsigned int m_currentRead;
	unsigned int m_currentWrite;

	unsigned int m_blockSize;
};


#include "Engine/cuda/NBody/bodysystemcuda_impl.h"