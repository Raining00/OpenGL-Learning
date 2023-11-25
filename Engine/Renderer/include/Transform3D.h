#pragma once

#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"

namespace Renderer
{
    class Transform3D
    {
        public:
            static const glm::vec3 LocalForward;
            static const glm::vec3 LocalUp;
            static const glm::vec3 LocalRight;

            Transform3D();
            ~Transform3D() = default;

            // GETTERS
            glm::mat4 getWorldMatrix();
            glm::mat3 getNormalMatrix();
            glm::mat4 getInvWorldMatrix();

            // Transformation
            void scale(const glm::vec3& scale);
            void translate(const glm::vec3& dt);
            void rotate(const glm::vec3& axis, float angle);
            void setScale(const glm::vec3& scale);
            void setRotation(const glm::quat& r);
            void setTranslation(const glm::vec3& t);

            // query for camera's axis
            glm::vec3 Forward() const;
            glm::vec3 Up() const;
            glm::vec3 Right() const;

            // getters
            glm::vec3 scale() const { return m_scale; }
            glm::quat rotation() const { return m_rotation; }
            glm::vec3 translation() const { return m_translation; }
        private:
            mutable bool m_dirty; // should update or not

            glm::vec3 m_scale;
            glm::quat m_rotation;
            glm::vec3 m_translation;
            glm::mat4 m_worldMatrix;
            glm::mat3 m_normalMatrix;
    };
}