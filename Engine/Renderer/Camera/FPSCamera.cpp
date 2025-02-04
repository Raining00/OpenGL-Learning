#include "FPSCamera.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include <iostream>
#ifdef _WIN32
#include "glad//glad.h"
#elif defined(__linux__)
#include <GL/glew.h>
#endif // _WIN32


namespace Renderer
{
    FPSCamera::FPSCamera(glm::vec3 pos)
        : m_dirty(true), m_translation(pos), m_rotation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f)), m_mouseSensitivity(0.05f), m_moveSpeed(2.5f)
    {
    }

    glm::mat4 FPSCamera::getViewMatrix()
    {
        if (m_dirty)
        {
            m_viewMatrix = glm::mat4(1.0);

            m_viewMatrix = glm::translate(glm::mat4_cast(glm::inverse(m_rotation)), -m_translation);
            m_dirty = false;
        }
        return m_viewMatrix;
    }

    glm::mat4 FPSCamera::getInvViewMatrix()
    {
        if (m_dirty)
        {
            m_dirty = false;
            m_viewMatrix = glm::mat4_cast(glm::conjugate(m_rotation));
            m_viewMatrix = glm::translate(m_viewMatrix, -m_translation);
        }
        return glm::inverse(m_viewMatrix);
    }

    void FPSCamera::onKeyPress(float deltaTime, char key)
    {
        switch (key)
        {
        case 'W':
            this->translate(this->getFront() * deltaTime * m_moveSpeed);
            break;
        case 'S':
            this->translate(-this->getFront() * deltaTime * m_moveSpeed);
            break;
        case 'A':
            this->translate(-this->getRight() * deltaTime * m_moveSpeed);
            break;
        case 'D':
            this->translate(this->getRight() * deltaTime * m_moveSpeed);
            break;
        case 'Q':
            this->translate(this->getUp() * deltaTime * m_moveSpeed);
            break;
        case 'E':
            this->translate(-this->getUp() * deltaTime * m_moveSpeed);
            break;
        default:
            break;
        }
    }

    void FPSCamera::onWheelMove(double delta)
    {
        // nothing now
    }

    void FPSCamera::onMouseMove(double deltaX, double deltaY, std::string button)
    {
        this->rotate(LocalUp, -deltaX * m_mouseSensitivity);
        this->rotate(getRight(), deltaY * m_mouseSensitivity);
    }

    void FPSCamera::lookAt(const glm::vec3& dir, const glm::vec3& up)
    {
        m_rotation = glm::quatLookAt(dir, up);
        m_dirty = true;
    }

    void FPSCamera::lookAt(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up)
    {
        glm::mat4 viewMatrix = glm::lookAt(eye, center, up);

        glm::mat3 rotationMat3(viewMatrix);
        m_rotation = glm::quat_cast(rotationMat3);

        m_translation = eye;

        m_dirty = true;
    }


    void FPSCamera::translate(const glm::vec3 &dt)
    {
        m_translation += dt;
        m_dirty = true;
    }

    void FPSCamera::rotate(const glm::vec3 &axis, float angle)
    {
        m_rotation = glm::angleAxis(glm::radians(angle), axis) * m_rotation;
        m_dirty = true;
    }

    void FPSCamera::setTranslation(const glm::vec3 &t)
    {
        m_translation = t;
        m_dirty = true;
    }

    void FPSCamera::setRotation(const glm::quat &r)
    {
        m_rotation = r;
        m_dirty = true;
    }

    glm::vec3 FPSCamera::getFront() const
    {
        return m_rotation * LocalForward;
    }

    glm::vec3 FPSCamera::getUp() const
    {
        return m_rotation * LocalUp;
    }

    glm::vec3 FPSCamera::getRight() const
    {
        return m_rotation * LocalRight;
    }

    void FPSCamera::updateMatrixUBO()
    {
        glBindBuffer(GL_UNIFORM_BUFFER, m_ubo);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(glm::mat4), glm::value_ptr(this->getProjectionMatrix()));
        glBufferSubData(GL_UNIFORM_BUFFER, sizeof(glm::mat4), sizeof(glm::mat4), glm::value_ptr(this->getViewMatrix()));
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
    }

}