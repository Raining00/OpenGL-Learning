#pragma once

#include <map>
#include <vector>

#include "Singleton.h"
#include "Light.h"

namespace Renderer{
/**
 * @brief A singleton class to manage all lights in the scene.
 * 
 * implemented light types:
 * 1. Directional Light
 * 2. Point Light
 * 3. Spot Light
 * You can use API to add light to the scene, and use API to set light uniform in shader.
 * 
 */
class LightManager : public Singleton<LightManager>
{
public:
    typedef std::shared_ptr<LightManager> ptr;
    LightManager() = default;
    ~LightManager() = default;

    static LightManager::ptr getInstance();

    /**
     * @brief Create a Directional Light object. A directional light is a light that has no position, only direction. Such as sun light.
     * 
     * @param name The name of the light.
     * @param direction Direction of the light.
     * @param ambient   Ambient color of the light.
     * @param diffuse   Diffuse color of the light.
     * @param specular  Specular color of the light.
     * @return unsigned int  The id of the light.(You can use this id to get the light with API getLight(unsigned int id)).
     * 
     * If you use the same name to add a light, nothing will happen and you will get the same id as the light you added before. 
     * make sure the name is unique especially when you create different types of light.
     * 
     */
    unsigned int CreateDirectionalLight(const std::string& name, const glm::vec3& direction, const glm::vec3& ambient, const glm::vec3& diffuse, const glm::vec3& specular);

    /**
     * @brief Create a Point Light object. A point light is a light that has position, and the light will be emitted in all directions. Such as light bulb.
     * 
     * @param name The name of the light.
     * @param position Position of the light.
     * @param ambient   Ambient color of the light.
     * @param diffuse   Diffuse color of the light.
     * @param specular  Specular color of the light.
     * @param constant  The constant attenuation of the light.
     * @param linear    The linear attenuation of the light.
     * @param quadratic The quadratic attenuation of the light.
     * @return unsigned int  The id of the light.(You can use this id to get the light with API getLight(unsigned int id)).
     * 
     * If you use the same name to add a light, nothing will happen and you will get the same id as the light you added before.
     * make sure the name is unique especially when you create different types of light
     * 
     */
    unsigned int CreatePointLight(const std::string& name, const glm::vec3& position, 
                                const glm::vec3& ambient, const glm::vec3& diffuse, const glm::vec3& specular, \
                                    float constant, float linear, float quadratic);

    /**
     * @brief Create a Spot Light object. A spot light is a light that has position and direction, and the light will be emitted in a cone. Such as flashlight.
     * 
     * @param name The name of the light.
     * @param position Position of the light.
     * @param direction Direction of the light.
     * @param ambient   Ambient color of the light.
     * @param diffuse   Diffuse color of the light.
     * @param specular  Specular color of the light.
     * @param constant  The constant attenuation of the light.
     * @param linear    The linear attenuation of the light.
     * @param quadratic The quadratic attenuation of the light.
     * @param innerCurOff The inner cut off angle of the light.
     * @param outerCutOff The outer cut off angle of the light.
     * @return unsigned int  The id of the light.(You can use this id to get the light with API getLight(unsigned int id)).
     * 
     * If you use the same name to add a light, nothing will happen and you will get the same id as the light you added before.
     * make sure the name is unique especially when you create different types of light
     * 
     */
    unsigned int CreateSpotLight(const std::string& name, const glm::vec3& position=glm::vec3(0.0, 0.0,0.0), const glm::vec3& direction=glm::vec3(0.0, 0.0, -1.0), 
                    const glm::vec3& ambient = glm::vec3(0.1), const glm::vec3& diffuse=glm::vec3(0.8), 
                    const glm::vec3& specular=glm::vec3(1.0), float constant=1.0, float linear=0.09, float quadratic=0.032, 
                    float cutOff=glm::cos(glm::radians(12.5f)), float outerCutOff= glm::cos(glm::radians(17.5f)));

    /**
     * @brief Get the Light object
     * 
     * @param name the name of the light you used when you create the light. Each light has a unique name.
     * @return Light::ptr 
     */
    Light::ptr getLight(const std::string& name);

    /**
     * @brief Get the Light object
     * 
     * @param id The id of the light returned by the create functions when you create the light. Each light has a unique id.
     * @return Light::ptr 
     */
    Light::ptr getLight(unsigned int id);

    /**
     * @brief Get the Light Index object
     * 
     * @param name The name of the light you used when you create the light. Each light has a unique name.
     * @return unsigned int The index of the light in the light array.
     * 
     */
    unsigned int getLightIndex(const std::string& name);

    /**
     * @brief Set the Light Uniform object
     * 
     * @param name The name of the light you used when you create the light.
     * @param shader Which shader you want to set the light uniform.
     * @param camera Which camera you want to set the light uniform.
     * @param lightName The name of the light uniform in the shader.
     * @param ifArray If the light uniform is an array.
     * @param slot The slot of the light uniform in the array.
     * 
     * You can both use the name or the id to set the light uniform.
     */
    void setLightUniform(const std::string& name, Shader::ptr shader, Camera3D::ptr camera, const std::string &lightName, const bool &ifArray=false, const unsigned int &slot=0);

    /**
     * @brief Set the Light Uniform object
     * 
     * @param id The id of the light returned by the create functions when you create the light. Each light has a unique id.
     * @param shader Which shader you want to set the light uniform.
     * @param camera Which camera you want to set the light uniform.
     * 
     * You can both use the name or the id to set the light uniform.
     */
    void setLightUniform(unsigned int id, Shader::ptr shader, Camera3D::ptr camera, const std::string &lightName, const bool &ifArray=false, const unsigned int &slot=0);
private:
    std::vector<Light::ptr> m_lights;
    std::map<std::string, unsigned int> m_lightMap;
};



}