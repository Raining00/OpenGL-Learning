#include "ShaderManager.h"

namespace Renderer
{

template<> ShaderManager::ptr Singleton<ShaderManager>::_instance = nullptr;

ShaderManager::ptr ShaderManager::getInstance()
{
    if (_instance == nullptr)
    {
        _instance = std::shared_ptr<ShaderManager>(new ShaderManager());
    }
    return _instance;

}

GLuint ShaderManager::loadShader(const std::string &shaderName, const std::string &vertexShaderPath, const std::string &fragmentShaderPath)
{
    if (m_shaderMap.find(shaderName) != m_shaderMap.end())
        return m_shaderMap[shaderName];

    std::shared_ptr<Shader> shader = std::make_shared<Shader>(vertexShaderPath.c_str(), fragmentShaderPath.c_str());
    m_shaders.push_back(shader);
    m_shaderMap[shaderName] = m_shaders.size() - 1;
    return m_shaders.size() - 1;
}

GLuint ShaderManager::loadShader(const std::string &shaderName, const std::string &vertexShaderPath, const std::string &fragmentShaderPath, const std::string &geometryShaderPath)
{
    if (m_shaderMap.find(shaderName) != m_shaderMap.end())
        return m_shaderMap[shaderName];

    std::shared_ptr<Shader> shader = std::make_shared<Shader>(vertexShaderPath.c_str(), fragmentShaderPath.c_str(), geometryShaderPath.c_str());
    m_shaders.push_back(shader);
    m_shaderMap[shaderName] = m_shaders.size() - 1;
    return m_shaders.size() - 1;
}

std::shared_ptr<Shader> ShaderManager::getShader(const std::string &shaderName)
{
    if (m_shaderMap.find(shaderName) == m_shaderMap.end())
        return nullptr;
    return m_shaders[m_shaderMap[shaderName]];
}

std::shared_ptr<Shader> ShaderManager::getShader(GLuint shaderID)
{
    if (shaderID >= m_shaders.size())
        return nullptr;
    return m_shaders[shaderID];
}

GLuint ShaderManager::getShdaerIndex(const std::string& shaderName)
{
    if (m_shaderMap.find(shaderName) == m_shaderMap.end())
        return 0;
    return m_shaderMap[shaderName];
}

bool ShaderManager::bindShader(GLuint shaderID)
{
    if (shaderID >= m_shaders.size())
        return false;
    m_shaders[shaderID]->use();
    return true;
}

bool ShaderManager::bindShader(const std::string &shaderName)
{
    if (m_shaderMap.find(shaderName) == m_shaderMap.end())
        return false;
    m_shaders[m_shaderMap[shaderName]]->use();
    return true;
}

bool ShaderManager::unbindShader()
{
    glUseProgram(0);
    return true;
}

}