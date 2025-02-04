#include "Camera3D.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#ifdef _WIN32
#include "glad/glad.h"
#elif defined(__linux__)
#include "GL/glew.h"
#endif 

namespace Renderer
{
    const glm::vec3 Camera3D::LocalForward(0.0f, 0.0f, -1.0f);
    const glm::vec3 Camera3D::LocalUp(0.0f, 1.0f, 0.0f);
    const glm::vec3 Camera3D::LocalRight(1.0f, 0.0f, 0.0f);

    void Camera3D::setAspect(float aspect)
    {
        m_aspect = aspect;
        m_projectionMatrix = glm::perspective(glm::radians(m_angle), m_aspect, m_near, m_far);
    }

    void Camera3D::setPerspective(float angle, float aspect, float near, float far)
    {
        m_angle = angle;
        m_aspect = aspect;
        m_near = near;
        m_far = far;
        m_projectionMatrix = glm::perspective(glm::radians(m_angle), m_aspect, m_near, m_far);
        m_invProjectionMatrix = glm::inverse(m_projectionMatrix);
    }

    void Camera3D::setOrtho(float left, float right, float bottom, float top, float near, float far)
    {
        m_projectionMatrix = glm::ortho(left, right, bottom, top, near, far);
        m_invProjectionMatrix = glm::inverse(m_projectionMatrix);
    }

    glm::mat4 Camera3D::getProjectionMatrix() const
    {
        return m_projectionMatrix;
    }

    glm::mat4 Camera3D::getInvProjectionMatrix() const
    {
        return m_invProjectionMatrix;
    }

    void Camera3D::createUBO()
    {
        glGenBuffers(1, &m_ubo);
        glBindBuffer(GL_UNIFORM_BUFFER, m_ubo);
        glBufferData(GL_UNIFORM_BUFFER, 2 * sizeof(glm::mat4), NULL, GL_STATIC_DRAW);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
        glBindBufferRange(GL_UNIFORM_BUFFER, 0, m_ubo, 0, 2 * sizeof(glm::mat4));
    }
} // namespace Renderer