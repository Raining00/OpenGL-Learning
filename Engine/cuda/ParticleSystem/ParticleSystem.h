#pragma once
#include <vector>
#include <memory>
#include "glm/glm.hpp"

namespace Physics
{
    class ParticleSystem
    {
    public:
        typedef std::shared_ptr<ParticleSystem> ptr;
        ParticleSystem() = default;
        ~ParticleSystem();

        bool initialize(unsigned int particleNum);
        bool reset();

        void setParticlePosition(const std::vector<glm::vec4>& hostPosition);
        void setParticlePosition(float* hostPosition, unsigned int particleNum);
        void setParticlePosition(const std::vector<float> hostPosition);

        void setParticleVelocity(const std::vector<glm::vec4>& hostVelocity);
        void setParticleVelocity(float* hostVelocity, unsigned int particleNum);

        unsigned int getPositionVBO() const { return m_PosVBO; }
    private:
        float* m_devPosition; // float4: (x, y, z, mass);
        float* m_devVelocity; // float4: (vs, vy, vz, 0.0)
        bool m_isInit{ false };

        std::vector<glm::vec4> m_hostPosition;
        std::vector<glm::vec4> m_hostVelocity;

        unsigned int m_particleNum{ 0 }, m_PosVBO;
        struct cudaGraphicsResource* m_cudaVBOResource;
    };
}
