#include "Light.h"

namespace Renderer
{
    void Light::setLightColor(glm::vec3 amb, glm::vec3 diff, glm::vec3 spec)
    {
        m_ambient = amb;
        m_diffuse = diff;
        m_specular = spec;
    }

    void DirectionalLight::setLightUniforms(Shader::ptr shader, Camera3D::ptr camera, const std::string &lightName, const bool &ifArray, const unsigned int &slot)
    {
        shader->setVec3("cameraPos", camera->getPosition());
        if (ifArray)
        {
            shader->setVec3(lightName + "[" + std::to_string(slot) + "].direction", m_direction);
            shader->setVec3(lightName + "[" + std::to_string(slot) + "].ambient", m_ambient);
            shader->setVec3(lightName + "[" + std::to_string(slot) + "].diffuse", m_diffuse);
            shader->setVec3(lightName + "[" + std::to_string(slot) + "].specular", m_specular);
        }
        else
        {
            shader->setVec3(lightName + ".direction", m_direction);
            shader->setVec3(lightName + ".ambient", m_ambient);
            shader->setVec3(lightName + ".diffuse", m_diffuse);
            shader->setVec3(lightName + ".specular", m_specular);
        }
    }

    void DirectionalLight::setLightDirection(glm::vec3 dir)
    {
        m_direction = dir;
    }

    glm::vec3 DirectionalLight::getDirection() const
    {
        return m_direction;
    }

    void PointLight::setLightUniforms(Shader::ptr shader, Camera3D::ptr camera, const std::string &lightName, const bool &ifArray, const unsigned int &slot)
    {
        shader->setVec3("cameraPos", camera->getPosition());
        if(ifArray)
        {
            shader->setVec3(lightName + "[" + std::to_string(slot) + "].position", m_position);
            shader->setVec3(lightName + "[" + std::to_string(slot) + "].ambient", m_ambient);
            shader->setVec3(lightName + "[" + std::to_string(slot) + "].diffuse", m_diffuse);
            shader->setVec3(lightName + "[" + std::to_string(slot) + "].specular", m_specular);
            shader->setFloat(lightName + "[" + std::to_string(slot) + "].constant", m_constant);
            shader->setFloat(lightName + "[" + std::to_string(slot) + "].linear", m_linear);
            shader->setFloat(lightName + "[" + std::to_string(slot) + "].quadratic", m_quadratic);
        }
        else
        {
            shader->setVec3(lightName + ".position", m_position);
            shader->setVec3(lightName + ".ambient", m_ambient);
            shader->setVec3(lightName + ".diffuse", m_diffuse);
            shader->setVec3(lightName + ".specular", m_specular);
            shader->setFloat(lightName + ".constant", m_constant);
            shader->setFloat(lightName + ".linear", m_linear);
            shader->setFloat(lightName + ".quadratic", m_quadratic);
        }
    }

    void PointLight::setLightPosition(glm::vec3 pos)
    {
        m_position = pos;
    }

    void PointLight::setLightAttenuation(float constant, double linear, double quadratic)
    {
        m_constant = constant;
        m_linear = linear;
        m_quadratic = quadratic;
    }

    void SpotLight::setLightUniforms(Shader::ptr shader, Camera3D::ptr camera, const std::string &lightName, const bool &ifArray, const unsigned int &slot)
    {
        shader->setVec3("cameraPos", camera->getPosition());
        if(ifArray)
        {
            shader->setVec3(lightName + "[" + std::to_string(slot) + "].position", m_position);
            shader->setVec3(lightName + "[" + std::to_string(slot) + "].direction", m_direction);
            shader->setVec3(lightName + "[" + std::to_string(slot) + "].ambient", m_ambient);
            shader->setVec3(lightName + "[" + std::to_string(slot) + "].diffuse", m_diffuse);
            shader->setVec3(lightName + "[" + std::to_string(slot) + "].specular", m_specular);
            shader->setFloat(lightName + "[" + std::to_string(slot) + "].constant", m_constant);
            shader->setFloat(lightName + "[" + std::to_string(slot) + "].linear", m_linear);
            shader->setFloat(lightName + "[" + std::to_string(slot) + "].quadratic", m_quadratic);
            shader->setFloat(lightName + "[" + std::to_string(slot) + "].innerCutOff", m_innerCutOff);
            shader->setFloat(lightName + "[" + std::to_string(slot) + "].outerCutOff", m_outerCutOff);
        }
        else
        {
            shader->setVec3(lightName + ".position", m_position);
            shader->setVec3(lightName + ".direction", m_direction);
            shader->setVec3(lightName + ".ambient", m_ambient);
            shader->setVec3(lightName + ".diffuse", m_diffuse);
            shader->setVec3(lightName + ".specular", m_specular);
            shader->setFloat(lightName + ".constant", m_constant);
            shader->setFloat(lightName + ".linear", m_linear);
            shader->setFloat(lightName + ".quadratic", m_quadratic);
            shader->setFloat(lightName + ".innerCutOff", m_innerCutOff);
            shader->setFloat(lightName + ".outerCutOff", m_outerCutOff);
        }
    }

    void SpotLight::setLightPosition(glm::vec3 pos)
    {
        m_position = pos;
    }

    void SpotLight::setLightDirection(glm::vec3 dir)
    {
        m_direction = dir;
    }

    void SpotLight::setLightAttenuation(float constant, float linear, float quadratic)
    {
        m_constant = constant;
        m_linear = linear;
        m_quadratic = quadratic;
    }

    void SpotLight::setLightCutoff(float innerCutOff, float outerCutOff)
    {
        m_innerCutOff = innerCutOff;
        m_outerCutOff = outerCutOff;
    }
}