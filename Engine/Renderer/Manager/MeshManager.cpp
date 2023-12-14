#include "MeshManager.h"

namespace Renderer
{
    template <>
    MeshManager::ptr Singleton<MeshManager>::_instance = nullptr;

    MeshManager::ptr MeshManager::getInstance()
    {
        if (_instance == nullptr)
            return _instance = std::make_shared<MeshManager>();
        return _instance;
    }

    unsigned int MeshManager::loadMesh(Mesh *mesh)
    {
        Mesh::ptr mptr(mesh);
        m_units.push_back(mptr);
        return m_units.size() - 1;
    }

    Mesh::ptr MeshManager::getMesh(unsigned int unit)
    {
        if (unit >= m_units.size())
            return nullptr;
        return m_units[unit];
    }

    bool MeshManager::drawMesh(unsigned int unit, bool instance, int amount)
    {
        if (unit >= m_units.size())
            return false;
        m_units[unit]->draw(instance, amount);
        return true;
    }
}