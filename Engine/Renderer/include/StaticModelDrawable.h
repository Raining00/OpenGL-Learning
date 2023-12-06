#pragma once

#include "../include/Drawable.h"
#include <map>
class aiMesh;
class aiNode;
class aiScene;

namespace Renderer
{
    class StaticModelDrawable : public Drawable
    {
    public:
        typedef std::shared_ptr<StaticModelDrawable> ptr;

        StaticModelDrawable(GLuint shaderIndex, const std::string& path);
        ~StaticModelDrawable() = default;

        virtual void getAABB(glm::vec3& min, glm::vec3& max) override { min = m_min; max = m_max; }

        virtual void render(Camera3D::ptr camera, Light::ptr sunLight, Camera3D::ptr lightCamera, Shader::ptr shader = nullptr) override;
        virtual void renderDepth(Shader::ptr shader, Camera3D::ptr lightCamera) override;

    protected:
        virtual void renderImp() override;
    
    private:
        void loadModel(const std::string& path);
        void processNode(aiNode* node, const aiScene* scene);
        void processMesh(aiMesh* mesh, const aiScene* scene, unsigned int& meshIndex, unsigned int& texIndex);

        void bind(int x);
        void unbind(int x);
    private:
        glm::vec3 m_min, m_max;
        std::string m_directory;
        // string: texture type: diffuse, specular, normal, height
        // unsigned int: texture index
        std::vector<std::map<std::string, unsigned int>> m_textureMapList;
    };
}