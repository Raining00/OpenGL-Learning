#pragma once

#include "glm/glm.hpp"
#include "Camera3D.h"
#include "Shader.h"

namespace Renderer
{
    class Light
    {
        public:
            typedef std::shared_ptr<Light> ptr;
            Light() = default;
            ~Light() = default;
            /**
             * @brief Set the Light Color object
             * 
             * @param amb ambient color
             * @param diff diffuse color
             * @param spec specular color
             */
            virtual void setLightColor(glm::vec3 amb, glm::vec3 diff, glm::vec3 spec);
            /**
             * @brief Set the Light Uniforms object
             * 
             * @param shader which shader to set the uniforms
             * @param camera which camera to set the uniforms
             */
            virtual void setLightUniforms(Shader::ptr shader, Camera3D::ptr camera) = 0;

            virtual glm::vec3 getAmbient() const { return m_ambient; }
            virtual glm::vec3 getDiffuse() const { return m_diffuse; }
            virtual glm::vec3 getSpecular() const { return m_specular; }
        protected:
            glm::vec3 m_ambient;
            glm::vec3 m_diffuse;
            glm::vec3 m_specular;
    };
    /**
     * @brief Directional light class
     * A directional light is a light that has no position, it is infinitely far away and has only a direction.
     * Such as the sun.
     */
    class DirectionalLight : public Light
    {
        public:
            typedef std::shared_ptr<DirectionalLight> ptr;
            DirectionalLight() = default;
            ~DirectionalLight() = default;

            /**
             * @brief Set the Light Uniforms object
             * The uniforms that will be set are:
             * cameraPos;
             * dirLight.direction;
             * dirLight.ambient;
             * dirLight.diffuse;
             * dirLight.specular;
             * make sure the shader has these uniforms. For example, in the fragment shader, you should define a struct like this:
             * struct DirLight
             * {
             *    vec3 direction;
             *    vec3 ambient;
             *    vec3 diffuse;
             *    vec3 specular;
             * };
             * uniform DirLight dirLight;
             * @param shader the light uniforms will be set to this shader, so the shader must have the correct uniforms.
             * @param camera 
             */
            void setLightUniforms(Shader::ptr shader, Camera3D::ptr camera) override;
            /**
             * @brief Set the Light Direction object
             * 
             * @param dir the direction of the light
             */
            void setLightDirection(glm::vec3 dir);

            glm::vec3 getDirection() const;
        private:
            glm::vec3 m_direction;
    };

    /**
     * @brief A point light is a light that has a position and shines in all directions.
     * Such as a light bulb.
     * 
     */
    class PointLight : public Light
    {
        public:
            typedef std::shared_ptr<PointLight> ptr;
            PointLight() = default;
            ~PointLight() = default;

            void setLightUniforms(Shader::ptr shader, Camera3D::ptr camera) override;
            void setLightPosition(glm::vec3 pos);
            void setLightAttenuation(float constant, float linear, float quadratic);
        private:
            glm::vec3 m_position;
            float m_constant;
            float m_linear;
            float m_quadratic;
    };

    /**
     * @brief A spot light is a light that has a position, direction and a cutoff.
     * the spot light will only shine in a cone shape. It always used to simulate a flashlight.
     * 
     */
    class SpotLight : public Light
    {
        public:
            typedef std::shared_ptr<SpotLight> ptr;
            SpotLight() = default;
            ~SpotLight() = default;

            void setLightUniforms(Shader::ptr shader, Camera3D::ptr camera) override;
            void setLightPosition(glm::vec3 pos);
            void setLightDirection(glm::vec3 dir);
            void setLightAttenuation(float constant, float linear, float quadratic);
            void setLightCutoff(float inner, float outer);
        private:
            glm::vec3 m_position;
            glm::vec3 m_direction;
            float m_constant;
            float m_linear;
            float m_quadratic;
            float m_innerCutoff;
            float m_outerCutoff;
    };
}