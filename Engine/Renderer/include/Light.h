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
     * @note the attenuation is calculated as:
     * attenuation = 1.0 / (constant + linear * distance + quadratic * (distance * distance));
     * 
     * By default, the attenuation is set to:
     * constant = 1.0f;
     * linear = 0.09f;
     * quadratic = 0.032f;
     * 
     * You can change the attenuation by calling setLightAttenuation(constant, linear, quadratic). According to this website:
     * https://learnopengl.com/Lighting/Light-casters
     */
    class PointLight : public Light
    {
    public:
        typedef std::shared_ptr<PointLight> ptr;
        PointLight() = default;
        ~PointLight() = default;
        /**
         * @brief Set the Light Uniforms object
         *      * The uniforms that will be set are:
         * pointLight.position;
         * pointLight.ambient;
         * pointLight.diffuse;
         * pointLight.specular;
         * pointLight.constant;
         * pointLight.linear;
         * pointLight.quadratic;
         * make sure the shader has these uniforms. For example, in the fragment shader, you should define a struct like this:
         * struct PointLight
         * {
         *   vec3 position;
         *   vec3 ambient;
         *   vec3 diffuse;
         *   vec3 specular;
         *   float constant;
         *   float linear;
         *   float quadratic;
         * };
         * uniform PointLight pointLight;
         *
         * @param shader the light uniforms will be set to this shader, so the shader must have the correct uniforms.
         * @param camera the camera uniforms will be set to this shader, so the shader must have the correct uniforms.
         */
        void setLightUniforms(Shader::ptr shader, Camera3D::ptr camera) override;
        void setLightPosition(glm::vec3 pos);
        void setLightAttenuation(float constant, double linear, double quadratic);
    public:
        //get methods
        glm::vec3 getPosition() const { return m_position; }
        float getConstant() const { return m_constant; }
        float getLinear() const { return m_linear; }
        float getQuadratic() const { return m_quadratic; }
        //get ptr methods (may be used in imgui to change the value)
        float* getConstantPtr() { return &m_constant; }
        float* getLinearPtr() { return &m_linear; }
        float* getQuadraticPtr() { return &m_quadratic; }
    private:
        glm::vec3 m_position;
        float m_constant{1.0f};
        float m_linear{0.09f};     // 0.09
        float m_quadratic{0.032f}; // 0.032
    };

    /**
     * @brief A spot light is a light that has a position, direction and a cutoff.
     * the spot light will only shine in a cone shape. It always used to simulate a flashlight.
     *
     * by default, the cutoff is set to:
     * innerCutoff = glm::cos(glm::radians(12.5f));
     * outerCutoff = glm::cos(glm::radians(17.5f));
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
        void setLightCutoff(float m_innerCutOff=glm::cos(glm::radians(12.5f)), float m_outerCutOff=glm::cos(glm::radians(17.5f)));
    public:
        //get method
        glm::vec3 getPosition() const { return m_position; }
        glm::vec3 getDirection() const { return m_direction; }
        float getConstant() const { return m_constant; }
        float getLinear() const { return m_linear; }
        float getQuadratic() const { return m_quadratic; }
        float getInnerCutoff() const { return m_innerCutOff; }
        float getOuterCutoff() const { return m_outerCutOff; }
        //get ptr methods (may be used in imgui to change the value)
        float* getConstantPtr() { return &m_constant; }
        float* getLinearPtr() { return &m_linear; }
        float* getQuadraticPtr() { return &m_quadratic; }
        float* getInnerCutoffPtr() { return &m_innerCutOff; }
        float* getOuterCutOffPtr() { return &m_outerCutOff; }
    private:
        glm::vec3 m_position{0.0f, 0.0f, 0.0f};
        glm::vec3 m_direction{0.0f, 0.0f, -1.0f};
        float m_constant{1.0f}; // 1.0
        float m_linear{0.09f};  // 0.09
        float m_quadratic{0.032f}; // 0.032
        float m_innerCutOff{glm::cos(glm::radians(12.5f))}; // 12.5
        float m_outerCutOff{glm::cos(glm::radians(17.5f))}; // 17.5
    };
}