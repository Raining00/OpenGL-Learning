#pragma once

#include <map>
#include <vector>

#include "Shader/Shader.h"
#include "Manager/Singleton.h"

namespace Renderer
{

class ShaderManager:public Singleton<ShaderManager>
{

public:
    typedef std::shared_ptr<ShaderManager> ptr;

    ShaderManager() = default;
    ~ShaderManager() = default;

    static ShaderManager::ptr getInstance();

    GLuint loadShader(const std::string &shaderName, const std::string &vertexShaderPath, const std::string &fragmentShaderPath);
    GLuint loadShader(const std::string &shaderName, const std::string &vertexShaderPath, const std::string &fragmentShaderPath, const std::string &geometryShaderPath);
    std::shared_ptr<Shader> getShader(const std::string &shaderName);

    std::shared_ptr<Shader> getShader(GLuint shaderID);

    bool bindShader(GLuint shaderID);

    bool bindShader(const std::string &shaderName);

    bool unbindShader();
private:
    std::vector<std::shared_ptr<Shader>> m_shaders;
    std::map<std::string, GLuint> m_shaderMap;

};


}