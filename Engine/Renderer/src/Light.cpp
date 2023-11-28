#include "Light.h"

namespace Renderer
{
    void Light::setLightColor(glm::vec3 amb, glm::vec3 diff, glm::vec3 spec)
    {
        m_ambient = amb;
        m_diffuse = diff;
        m_specular = spec;
    }

    void DirectionalLight::setLightUniforms(Shader::ptr shader, Camera3D::ptr camera)
    {
        shader->setVec3("cameraPos", camera->getPosition());
        shader->setVec3("dirLight.direction", m_direction);
        shader->setVec3("dirLight.ambient", m_ambient);
        shader->setVec3("dirLight.diffuse", m_diffuse);
        shader->setVec3("dirLight.specular", m_specular);
    }

    void DirectionalLight::setLightDirection(glm::vec3 dir)
    {
        m_direction = dir;
    }

    glm::vec3 DirectionalLight::getDirection() const
    {
        return m_direction;
    }

    void PointLight::setLightUniforms(Shader::ptr shader, Camera3D::ptr camera)
    {
        shader->setVec3("cameraPos", camera->getPosition());
        shader->setVec3("pointLight.position", m_position);
        shader->setVec3("pointLight.ambient", m_ambient);
        shader->setVec3("pointLight.diffuse", m_diffuse);
        shader->setVec3("pointLight.specular", m_specular);
        shader->setFloat("pointLight.constant", m_constant);
        shader->setFloat("pointLight.linear", m_linear);
        shader->setFloat("pointLight.quadratic", m_quadratic);
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

    void SpotLight::setLightUniforms(Shader::ptr shader, Camera3D::ptr camera)
    {
        shader->setVec3("cameraPos", camera->getPosition());
        shader->setVec3("spotLight.position", m_position);
        shader->setVec3("spotLight.direction", m_direction);
        shader->setVec3("spotLight.ambient", m_ambient);
        shader->setVec3("spotLight.diffuse", m_diffuse);
        shader->setVec3("spotLight.specular", m_specular);
        shader->setFloat("spotLight.constant", m_constant);
        shader->setFloat("spotLight.linear", m_linear);
        shader->setFloat("spotLight.quadratic", m_quadratic);
        shader->setFloat("spotLight.innerCutOff", m_innerCutOff);
        shader->setFloat("spotLight.outerCutOff", m_outerCutOff);
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