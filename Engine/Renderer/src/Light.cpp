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

    void PointLight::setLightAttenuation(float constant, float linear, float quadratic)
    {
        m_constant = constant;
        m_linear = linear;
        m_quadratic = quadratic;
    }

}