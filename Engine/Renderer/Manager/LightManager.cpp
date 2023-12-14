#include "LightManager.h"
#include "../../include/ColorfulPrint.h"

namespace Renderer
{

template <>
LightManager::ptr Singleton<LightManager>::_instance = nullptr;

LightManager::ptr LightManager::getInstance()
{
    if (_instance == nullptr)
    {
        _instance = std::make_shared<LightManager>();
    }
    return _instance;
}

unsigned int LightManager::CreateDirectionalLight(const std::string& name, const glm::vec3& direction, const glm::vec3& ambient, const glm::vec3& diffuse, const glm::vec3& specular)
{
    if (m_lightMap.find(name) != m_lightMap.end())
        return m_lightMap[name];
    Light::ptr light = std::make_shared<DirectionalLight>();
    light->setLightColor(ambient, diffuse, specular);
    DirectionalLight* dirLight = reinterpret_cast<DirectionalLight*>(light.get());
    dirLight->setLightDirection(direction);

    m_lights.push_back(light);
    m_lightMap[name] = m_lights.size() - 1;
    return m_lights.size() - 1;
}

unsigned int LightManager::CreatePointLight(const std::string& name, const glm::vec3& position, const glm::vec3& ambient, const glm::vec3& diffuse, const glm::vec3& specular, float constant, float linear, float quadratic)
{
    if (m_lightMap.find(name) != m_lightMap.end())
        return m_lightMap[name];
    Light::ptr light = std::make_shared<PointLight>();
    light->setLightColor(ambient, diffuse, specular);
    PointLight* pointLight = reinterpret_cast<PointLight*>(light.get());
    pointLight->setLightPosition(position);
    pointLight->setLightAttenuation(constant, linear, quadratic);

    m_lights.push_back(light);
    m_lightMap[name] = m_lights.size() - 1;
    return m_lights.size() - 1;
}

unsigned int LightManager::CreateSpotLight(const std::string& name, const glm::vec3& position, const glm::vec3& direction, 
                    const glm::vec3& ambient, const glm::vec3& diffuse, 
                    const glm::vec3& specular, float constant, float linear, float quadratic, 
                    float cutOff, float outerCutOff)
{
    if (m_lightMap.find(name) != m_lightMap.end())
        return m_lightMap[name];
    Light::ptr light = std::make_shared<SpotLight>();
    light->setLightColor(ambient, diffuse, specular);
    SpotLight* spotLight = reinterpret_cast<SpotLight*>(light.get());
    spotLight->setLightPosition(position);
    spotLight->setLightDirection(direction);
    spotLight->setLightAttenuation(constant, linear, quadratic);
    spotLight->setLightCutoff(cutOff, outerCutOff);

    m_lights.push_back(light);
    m_lightMap[name] = m_lights.size() - 1;
    return m_lights.size() - 1;
}

Light::ptr LightManager::getLight(unsigned int id)
{
    if (id >= m_lights.size())
    {
        PRINT_ERROR("LightManager::getLight: id out of range");
        return nullptr;
    }
    return m_lights[id];
}

Light::ptr LightManager::getLight(const std::string& name)
{
    if (m_lightMap.find(name) != m_lightMap.end())
        return m_lights[m_lightMap[name]];
    else
        return nullptr;
}

unsigned int LightManager::getLightIndex(const std::string& name)
{
    if (m_lightMap.find(name) != m_lightMap.end())
        return m_lightMap[name];
    else
        return -1;
}

void LightManager::setLightUniform(const std::string& name, Shader::ptr shader, Camera3D::ptr camera, const std::string &lightName, const bool &ifArray, const unsigned int &slot)
{
    if (m_lightMap.find(name) != m_lightMap.end())
    {
        m_lights[m_lightMap[name]]->setLightUniforms(shader, camera, lightName, ifArray, slot);
    }
}

void LightManager::setLightUniform(unsigned int id, Shader::ptr shader, Camera3D::ptr camera, const std::string &lightName, const bool &ifArray, const unsigned int &slot)
{
    if (id < m_lights.size())
    {
        m_lights[id]->setLightUniforms(shader, camera, lightName, ifArray, slot);
    }
}

void LightManager::setLight(Shader::ptr shader, Camera3D::ptr camera)
{
    unsigned int pointLightNum = 0;
    for (auto& light : m_lightMap)
    {
	    if(light.first.find("DirectionalLight") != std::string::npos)
			m_lights[light.second]->setLightUniforms(shader, camera, "directionalLight", false);
        else if (light.first.find("SunLight") != std::string::npos)
            m_lights[light.second]->setLightUniforms(shader, camera, "sunLight", false, 0);
        else if (light.first.find("PointLight") != std::string::npos)
        {
            m_lights[light.second]->setLightUniforms(shader, camera, "pointLights",  true, pointLightNum++);
        }
		else if(light.first.find("SpotLight") != std::string::npos)
			m_lights[light.second]->setLightUniforms(shader, camera, "spotLight", false);
		else
			PRINT_ERROR("LightManager::setLight: light type not supported")
	}
    shader->setInt("pointLightNum", pointLightNum);
}

} // namespace Renderer

