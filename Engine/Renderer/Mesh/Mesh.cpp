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

void Mesh::setupMesh(const std::vector<Vertex>& vert, const std::vector<unsigned int>& ind)
{
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

