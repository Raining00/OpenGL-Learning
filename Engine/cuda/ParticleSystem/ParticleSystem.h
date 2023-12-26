#pragma once
#include <vector>

#include "glm/glm.hpp"

namespace Physics
{
    class ParticleSystem
    {
    public:
        ParticleSystem() = default;
        ~ParticleSystem();

        bool initialize();
        bool reset();

        void setParticlePosition(std::vector<glm::vec4> hostPosition);
        void setParticlePosition(std::vector<float> hostPosition);
        void setParticleVelocity(std::vector<glm::vec4> hostVelocity);
        void setParticleVelocity(std::vector<float> hostVelocity);

    private:
        float* m_devPosition; // float4: (x, y, z, mass);
        float* m_devVelocity; // float4: (vs, vy, vz, 0.0)

        std::vector<glm::vec4> m_hostPosition;
        std::vector<glm::vec4> hostVelocity;

        unsigned int m_particleNum, m_particleVBO;
        struct cudaGraphicsResource* m_cudaVBOResource;
    };
}