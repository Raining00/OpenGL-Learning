#pragma once

#include "Mesh/Mesh.h"
#include "Manager/Singleton.h"

#include <map>
#include <vector>

namespace Renderer
{
    class MeshManager : public Singleton<MeshManager>
    {
    private:
        std::vector<Mesh::ptr> m_units;

    public:
        typedef std::shared_ptr<MeshManager> ptr;

        static MeshManager::ptr getInstance();

        unsigned int loadMesh(Mesh *mesh);

        Mesh::ptr getMesh(unsigned int unit);

        bool drawMesh(unsigned int unit, bool instance, int amount = 0);
    };
}