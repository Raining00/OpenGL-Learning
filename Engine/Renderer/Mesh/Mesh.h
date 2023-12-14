#pragma once

#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace Renderer
{

    struct Vertex
    {
    public:
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 texCoords;
        glm::vec3 color;

        Vertex() = default;
        Vertex(float px, float py, float pz, float nx, float ny, float nz, float u, float v, float r, float g, float b)
        {
            position = glm::vec3(px, py, pz);
            normal = glm::vec3(nx, ny, nz);
            texCoords = glm::vec2(u, v);
            color = glm::vec3(r, g, b);
        }
    };

    class Mesh
    {
    public:
        typedef std::shared_ptr<Mesh> ptr;
        Mesh() = default;
        Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices);
        virtual ~Mesh();

        void setAABB(glm::vec3 min, glm::vec3 max)
        {
            m_minVertex = min;
            m_maxVertex = max;
        }
        bool getAABB(glm::vec3& min, glm::vec3& max)
        {
            if (m_minVertex == glm::vec3(0.0f) && m_maxVertex == glm::vec3(0.0f))
                return false;
            min = m_minVertex;
            max = m_maxVertex;
            return true;
        }

        unsigned int getVAO() const { return m_vao; }
        unsigned int getVBO() const { return m_vbo; }
        unsigned int getIndexBuffer() const { return m_ebo; }

        void draw(bool instance, int amount = 0) const;

    protected:
        void setupMesh(const std::vector<Vertex>& vertices, const std::vector<unsigned int>& indices);
        void clearMesh();
    private:
        unsigned int m_vao, m_vbo, m_ebo;
        std::vector<Vertex> m_vertices;
        std::vector<unsigned int> m_indices;
        glm::vec3 m_minVertex, m_maxVertex;
    };

}