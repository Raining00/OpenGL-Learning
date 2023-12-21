#include "Mesh.h"

#ifdef _WIN32
#include <glad/glad.h>
#elif defined(__linux__)
#include "GL/glew.h"
#endif

namespace Renderer
{

Mesh::Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices)
{
    setupMesh(vertices, indices);
}

Mesh::~Mesh()
{
    clearMesh();
}

void Mesh::draw(bool instance, int amount) const
{
    glBindVertexArray(m_vao);
    if (instance)
    {
        glDrawElementsInstanced(GL_TRIANGLES, m_indices.size(), GL_UNSIGNED_INT, 0, amount);
    }
    else
    {
        glDrawElements(GL_TRIANGLES, m_indices.size(), GL_UNSIGNED_INT, 0);
    }
    glBindVertexArray(0);
}

void Mesh::setupMesh(std::vector<Vertex>& vert, std::vector<unsigned int>& ind)
{
    // calculate tangent, the bitangent is the cross product of the normal and the tangent and we can calculate it in the shader, I will not do it here
    // calculate tangent for each triangle and then average it.
    for (int i = 0; i < ind.size(); i += 3)
    {
        Vertex& v0 = vert[ind[i]];
        Vertex& v1 = vert[ind[i + 1]];
        Vertex& v2 = vert[ind[i + 2]];

        glm::vec3 edge1 = v1.position - v0.position;
        glm::vec3 edge2 = v2.position - v0.position;
        glm::vec2 deltaUV1 = v1.texCoords - v0.texCoords;
        glm::vec2 deltaUV2 = v2.texCoords - v0.texCoords;

        float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

        glm::vec3 tangent;
        tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
        tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
        tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
        tangent = glm::normalize(tangent);

        v0.tangent += tangent;
        v1.tangent += tangent;
        v2.tangent += tangent;
    }

    for (Vertex& vertex : vert)
    {
        vertex.tangent = glm::normalize(vertex.tangent);
    }

    m_vertices = vert;
    m_indices = ind;

    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);
    glGenBuffers(1, &m_ebo);

    glBindVertexArray(m_vao);

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, m_vertices.size() * sizeof(Vertex), &m_vertices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_indices.size() * sizeof(unsigned int), &m_indices[0], GL_STATIC_DRAW);

    // Vertex Positions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);

    // Vertex Normals
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));

    // Vertex Texture Coords
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoords));

    // Vertex Color
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, color));

    // Vertex Tangent
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, tangent));

    glBindVertexArray(0);
}


void Mesh::clearMesh()
{
    std::vector<Vertex>().swap(m_vertices);
    std::vector<unsigned int>().swap(m_indices);
    glDeleteVertexArrays(1, &m_vao);
    glDeleteBuffers(1, &m_vbo);
    glDeleteBuffers(1, &m_ebo);
}

} // namespace Renderer

