#include "TextureManager.h"
#include "../../include/ColorfulPrint.h"

namespace Renderer
{
    template <>
    TextureManager::ptr Singleton<TextureManager>::_instance = nullptr;

    TextureManager::ptr TextureManager::getInstance()
    {
        if (_instance == nullptr)
        {
            _instance = std::make_shared<TextureManager>();
        }
        return _instance;
    }

    unsigned int TextureManager::loadTexture2D(const std::string &name, const std::string &path, glm::vec4 bColor, const TextureType& textureType)
    {
        if (m_textureMap.find(name) != m_textureMap.end())
            return m_textureMap[name];
        Texture::ptr texture = std::make_shared<Texture2D>(path, bColor, textureType);
        m_textures.push_back(texture);
        m_textureMap[name] = m_textures.size() - 1;
        return m_textures.size() - 1;
    }

    unsigned int TextureManager::loadTexture2D(const std::string &name, unsigned char *images, int width, int height, int channels, const TextureType& textureType)
    {
        if (m_textureMap.find(name) != m_textureMap.end())
            return m_textureMap[name];
        Texture::ptr texture = std::make_shared<Texture2D>(images, width, height, channels, textureType);
        m_textures.push_back(texture);
        m_textureMap[name] = m_textures.size() - 1;
        return m_textures.size() - 1;
    }

    unsigned int TextureManager::loadTextureCube(const std::string &name, const std::string &path, const std::string& posFix)
    {
        if (m_textureMap.find(name) != m_textureMap.end())
            return m_textureMap[name];
        Texture::ptr texture = std::make_shared<TextureCube>(path, posFix);
        m_textures.push_back(texture);
        m_textureMap[name] = m_textures.size() - 1;
        return m_textures.size() - 1;
    }

    unsigned int TextureManager::loadTextureDepth(const std::string &name, int width, int height)
    {
        if (m_textureMap.find(name) != m_textureMap.end())
            return m_textureMap[name];
        Texture::ptr texture = std::make_shared<TextureDepth>(width, height);
        m_textures.push_back(texture);
        m_textureMap[name] = m_textures.size() - 1;
        return m_textures.size() - 1;
    }

    unsigned int TextureManager::loadTextureStencil(const std::string& name, int width, int height)
    {
        if (m_textureMap.find(name) != m_textureMap.end())
            return m_textureMap[name];
        Texture::ptr texture = std::make_shared<TextureStencil>(width, height);
        m_textures.push_back(texture);
        m_textureMap[name] = m_textures.size() - 1;
        return m_textures.size() - 1;
    }

    unsigned int TextureManager::loadTextureDepthStencil(const std::string& name, int width, int height)
    {
		if (m_textureMap.find(name) != m_textureMap.end())
			return m_textureMap[name];
		Texture::ptr texture = std::make_shared<TextureDepthStencil>(width, height);
		m_textures.push_back(texture);
		m_textureMap[name] = m_textures.size() - 1;
		return m_textures.size() - 1;
    }

    unsigned TextureManager::loadTextureColor(const std::string &name, int width, int height, bool hdr)
    {
        if (m_textureMap.find(name) != m_textureMap.end())
            return m_textureMap[name];
        Texture::ptr texture = std::make_shared<TextureColor>(width, height, hdr);
        m_textures.push_back(texture);
        m_textureMap[name] = m_textures.size() - 1;
        return m_textures.size() - 1;
    }

    Texture::ptr TextureManager::getTexture(const std::string &name)
    {
        if (m_textureMap.find(name) != m_textureMap.end())
            return m_textures[m_textureMap[name]];
        else
            return nullptr;
    }

    Texture::ptr TextureManager::getTexture(unsigned int id)
    {
        if (id < m_textures.size())
            return m_textures[id];
        else
            return nullptr;
    }

    unsigned int TextureManager::getTextureIndex(const std::string &name)
    {
        if (m_textureMap.find(name) != m_textureMap.end())
            return m_textureMap[name];
        else
            return -1;
    }

    bool TextureManager::bindTexture(const std::string &name, unsigned int slot)
    {
        if (m_textureMap.find(name) != m_textureMap.end())
        {
            m_textures[m_textureMap[name]]->bind(slot);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool TextureManager::bindTexture(unsigned int id, unsigned int slot)
    {
        if (id < m_textures.size())
        {
            m_textures[id]->bind(slot);
            return true;
        }
        else
        {
            return false;
        }
    }

    void TextureManager::unbindTexture(const std::string &name)
    {
        if (m_textureMap.find(name) != m_textureMap.end())
        {
            m_textures[m_textureMap[name]]->unbind();
        }
    }

    void TextureManager::unbindTexture(unsigned int id)
    {
        if (id < m_textures.size())
        {
            m_textures[id]->unbind();
        }
    }
}