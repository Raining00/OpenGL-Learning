#pragma once

#include <string>
#include <memory>

#include "glm/glm.hpp"

namespace Renderer
{

    class Camera3D
    {
    public:
        typedef std::shared_ptr<Camera3D> ptr;
        // local axis
        static const glm::vec3 LocalForward;
        static const glm::vec3 LocalUp;
        static const glm::vec3 LocalRight;

        Camera3D() = default;
        virtual ~Camera3D() = default;

        void createUBO();
        virtual void updateMatrixUBO() = 0;

        // setting functions
        void setAspect(float aspect);
        void setPerspective(float angle, float aspect, float near, float far);
        void setOrtho(float left, float right, float bottom, float top, float near, float far);

        // getting functions
        glm::mat4 getProjectionMatrix() const;
        glm::mat4 getInvProjectionMatrix() const;
        float getFovy() const { return m_angle; }
        float getAspect() const { return m_aspect; }
        float getNear() const { return m_near; }
        float getFar() const { return m_far; }

        virtual glm::vec3 getPosition() = 0;
        virtual glm::mat4 getViewMatrix() = 0;
        virtual glm::mat4 getInvViewMatrix() = 0;

        // keyboard and mouse control
        virtual void onKeyPress(float deltaTime, char key) = 0;
        virtual void onWheelMove(double delta) = 0;
        virtual void onMouseMove(double deltaX, double deltaY, std::string button) = 0;

        virtual glm::vec3  getFront() const = 0;
        virtual glm::vec3  getRight() const = 0;
        virtual glm::vec3  getUp() const = 0;
    protected:
        glm::mat4 m_viewMatrix{glm::mat4(1.0f)};
        glm::mat4 m_projectionMatrix{glm::mat4(1.0f)};
        glm::mat4 m_invProjectionMatrix{glm::mat4(1.0f)};
        float m_angle, m_aspect, m_near, m_far;
        unsigned int m_ubo;
    };

}