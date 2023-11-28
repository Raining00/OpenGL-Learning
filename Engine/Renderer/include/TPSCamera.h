#pragma once

#include "Camera3D.h"
#include "Transform3D.h"

namespace Renderer
{
    class TPSCamera : public Camera3D
    {
    public:
        typedef std::shared_ptr<TPSCamera> ptr;

        TPSCamera(glm::vec3 target, float yaw, float pitch, float dist);
        virtual ~TPSCamera() = default;

        //getter
        glm::mat4 getTargetMatrix();
        virtual glm::vec3 getPosition() override;
        virtual glm::mat4 getViewMatrix() override;
        virtual glm::mat4 getInvViewMatrix() override;

        // setter
        void setYaw(float yaw) { m_yaw = yaw;}
        void setPitch(float pitch) { m_pitch = pitch; }
        void setDistance(float dist) { m_distance = dist; }
        void setMoveSpeed(float speed) { m_moveSpeed = speed; }
        void setMouseSensitivity(float sensitivity) { m_mouseSty = sensitivity; }
        void setWheelSensitivity(float sensitivity) { m_wheelSty = sensitivity; }
        void setDistanceLimit(float min, float max) { m_distanceLimt = glm::vec2(min, max); }

        // keyboard and mouse input
        virtual void onKeyPress(float deltaTime, char key) override;
        virtual void onWheelMove(double delta) override;
        virtual void onMouseMove(double deltaX, double deltaY, std::string button) override;

        glm::vec3 getFront() const override;
        glm::vec3 getRight() const override;
        glm::vec3 getUp() const override;
    private:
        mutable bool m_dirty; // should update or not
        glm::vec3 m_cameraPos;
        Transform3D m_target;
        glm::vec2 m_distanceLimt;
        double m_yaw, m_pitch, m_distance;
        float m_mouseSty, m_moveSpeed, m_wheelSty;
        
    private:
        void update();
    };
}