#pragma once

#include <map>
#include <vector>

#include "Singleton.h"
#include "Texture/Texture.h"

namespace Renderer
{

    class TextureManager : public Singleton<TextureManager>
    {
    public:
        typedef std::shared_ptr<TextureManager> ptr;
        TextureManager() = default;
        ~TextureManager() = default;

        static TextureManager::ptr getInstance();

        unsigned int loadTexture2D(const std::string& name, const std::string& path, glm::vec4 bColor = glm::vec4(1.0f), const TextureType& textureType = TextureType::AMBIENT);
        unsigned int loadTexture2D(const std::string& name, unsigned char* images, int width, int height, int channels, const TextureType& textureType = TextureType::AMBIENT);

        unsigned int loadTextureCube(const std::string& name, const std::string& path, const std::string& posFix);

        unsigned int loadTextureDepth(const std::string& name, int width, int height);

        unsigned int loadTextureStencil(const std::string& name, int width, int height);

        unsigned int loadTextureDepthStencil(const std::string& name, int width, int height);

        unsigned loadTextureColor(const std::string& name, int width, int height, bool hdr = false);

        Texture::ptr getTexture(const std::string& name);

        Texture::ptr getTexture(unsigned int id);

        unsigned int getTextureIndex(const std::string& name);

        bool bindTexture(const std::string& name, unsigned int slot = 0);
        bool bindTexture(unsigned int id, unsigned int slot = 0);

        void unbindTexture(const std::string& name);
        void unbindTexture(unsigned int id);

    private:
        std::vector<Texture::ptr> m_textures;
        std::map<std::string, unsigned int> m_textureMap;
    };
}
