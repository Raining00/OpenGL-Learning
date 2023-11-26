#pragma once

#include "Camera3D.h"
#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"

namespace Renderer
{

    class FPSCamera : public Camera3D
    {
    public:
        typedef std::shared_ptr<FPSCamera> ptr;
        FPSCamera(glm::vec3 pos);
        virtual ~FPSCamera() = default;

        virtual glm::vec3 getPosition() { return m_translation; }
        virtual glm::mat4 getViewMatrix() override;
        virtual glm::mat4 getInvViewMatrix() override;

        //setting functions
        void setMouseSensitivity(float sensitivity) { m_mouseSensitivity = sensitivity; }
        void setMoveSpeed(float speed) { m_moveSpeed = speed; }

        // keyboard and mouse input
        virtual void onKeyPress(float deltaTime, char key) override;
        virtual void onWheelMove(double delta) override;
        virtual void onMouseMove(double deltaX, double deltaY, std::string button) override;

        // Transform camera's axis
        void lookAt(glm::vec3 dir, glm::vec3 up);
        void translate(const glm::vec3& dt);
        void rotate(const glm::vec3& axis, float angle);
        void setTranslation(const glm::vec3& t);
        void setRotation(const glm::quat& r);

        // query for camera's axis
        glm::vec3 Forward() const;
        glm::vec3 Up() const;
        glm::vec3 Right() const;
    private:
        mutable bool m_dirty; // should update or not
        glm::vec3 m_translation;
        glm::quat m_rotation;
        float m_mouseSensitivity, m_moveSpeed;
    };

}