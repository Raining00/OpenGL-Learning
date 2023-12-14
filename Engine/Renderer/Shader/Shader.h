#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#ifdef _WIN32
#include "glad/glad.h"
#elif defined(__linux__)
#include <GL/glew.h>
#endif
#include <glm/glm.hpp>
#include <memory>
namespace Renderer{
class Shader
{
public:
    typedef std::shared_ptr<Shader> ptr;
    Shader(const char* vertexPath, const char* fragmentPath);
    Shader(const char* vertexPath, const char* fragmentPath, const char* geometryPath);
    Shader(const std::string & vertexPath, const std::string & fragmentPath);
    Shader(const std::string & vertexPath, const std::string & fragmentPath, const std::string & geometryPath);
    ~Shader();

    void use();

public:

    void setBool(const std::string &name, bool value) const;
    void setInt(const std::string &name, int value) const;
    void setFloat(const std::string &name, float value) const;
    void setDouble(const std::string &name, double value) const;

    void setVec2(const std::string &name, float x, float y) const;
    void setVec2(const std::string &name, glm::vec2 value) const;

    void setVec3(const std::string &name, float x, float y, float z) const;
    void setVec3(const std::string &name, glm::vec3 value) const;

    void setVec4(const std::string &name, float x, float y, float z, float w) const;
    void setVec4(const std::string &name, glm::vec4 value) const;

    void setMat2(const std::string &name, float *value) const;
    void setMat2(const std::string &name, glm::mat2 value) const;

    void setMat3(const std::string &name, float *value) const;
    void setMat3(const std::string &name, glm::mat3 value) const;

    void setMat4(const std::string &name, float *value) const;
    void setMat4(const std::string &name, glm::mat4 value) const;

private:
    unsigned int ID;
    void checkCompileErrors(unsigned int shader, std::string type);
};


}