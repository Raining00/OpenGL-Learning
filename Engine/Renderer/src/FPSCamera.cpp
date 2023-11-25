#include "FPSCamera.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include <iostream>

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
            m_dirty = false;
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
            this->translate(this->Forward() * deltaTime * m_moveSpeed);
            break;
        case 'S':
            this->translate(-this->Forward() * deltaTime * m_moveSpeed);
            break;
        case 'A':
            this->translate(-this->Right() * deltaTime * m_moveSpeed);
            break;
        case 'D':
            this->translate(this->Right() * deltaTime * m_moveSpeed);
            break;
        case 'Q':
            this->translate(this->Up() * deltaTime * m_moveSpeed);
            break;
        case 'E':
            this->translate(-this->Up() * deltaTime * m_moveSpeed);
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
        this->rotate(Right(), deltaY * m_mouseSensitivity);
    }

    void FPSCamera::lookAt(glm::vec3 dir, glm::vec3 up)
    {
        m_rotation = glm::quatLookAt(dir, up);
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

    glm::vec3 FPSCamera::Forward() const
    {
        return m_rotation * LocalForward;
    }

    glm::vec3 FPSCamera::Up() const
    {
        return m_rotation * LocalUp;
    }

    glm::vec3 FPSCamera::Right() const
    {
        return m_rotation * LocalRight;
    }
}