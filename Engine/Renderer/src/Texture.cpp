#include "Texture.h"
#include <vector>

#include <GL/glew.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "ColorfulPrint.h"

namespace Renderer
{
    Texture2D::Texture2D(unsigned char *images, int width, int height, int channels) : m_width(width), m_height(height), m_channels(channels)
    {
        glGenTextures(1, &m_id);
        glBindTexture(GL_TEXTURE_2D, m_id);
        // filter setting
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // minification filter
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // magnification filter
        // wrap setting
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); // wrap on x axis
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT); // wrap on y axis
        // load the image
        GLenum format;
        if (channels == 1)
            format = GL_RED;
        else if (channels == 3)
            format = GL_RGB;
        else if (channels == 4)
            format = GL_RGBA;
        else
            PRINT_ERROR("Unsupported image format!");
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, images);
        glGenerateMipmap(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    Texture2D::Texture2D(const std::string &path, glm::vec4 bColor) : m_borderColor(bColor)
    {
        setupTexture(path, "");
    }

    Texture2D::~Texture2D()
    {
        clearTexture();
    }

    void Texture2D::bind(unsigned int slot)
    {
        glActiveTexture(GL_TEXTURE0 + slot);
        glBindTexture(GL_TEXTURE_2D, m_id);
    }

    void Texture2D::unbind()
    {
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void Texture2D::setupTexture(const std::string &path, const std::string &pFix)
    {
        // texture unit generate
        glGenTextures(1, &m_id);
        glBindTexture(GL_TEXTURE_2D, m_id);
        // filter setting
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // minification filter
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // magnification filter
        // wrap setting
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); // wrap on x axis
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT); // wrap on y axis
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, &m_borderColor[0]);
        // load the image
        int width, height, channels;
        // rotate the image
        stbi_set_flip_vertically_on_load(true);
        unsigned char *data = stbi_load((path + pFix).c_str(), &width, &height, &channels, 0);
        if (data)
        {
            GLenum format;
            if (channels == 1)
                format = GL_RED;
            else if (channels == 3)
                format = GL_RGB;
            else if (channels == 4)
                format = GL_RGBA;
            else
                PRINT_ERROR("Unsupported image format!");
            glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D);
        }
        else
            PRINT_ERROR("Failed to load texture!");

        stbi_image_free(data);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void Texture2D::clearTexture()
    {
        glDeleteTextures(1, &m_id);
    }

    TextureCube::TextureCube(const std::string &path, const std::string *posFix)
    {
        setupTexture(path, *posFix);
    }

    TextureCube::~TextureCube()
    {
        clearTexture();
    }

    void TextureCube::bind(unsigned slot)
    {
        glActiveTexture(GL_TEXTURE0 + slot);
        glBindTexture(GL_TEXTURE_CUBE_MAP, m_id);
    }

    void TextureCube::unbind()
    {
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    }

    void TextureCube::setupTexture(const std::string &path, const std::string &pFix)
    {
        // load cube map
        glGenTextures(1, &m_id);
        glBindTexture(GL_TEXTURE_CUBE_MAP, m_id);
        int width, height, nrComponents;
        // 6 faces.
        std::vector<std::string> faces = {
            path + "right" + pFix,
            path + "left" + pFix,
            path + "top" + pFix,
            path + "bottom" + pFix,
            path + "front" + pFix,
            path + "back" + pFix};
        // load image
        for (int x = 0; x < faces.size(); x++)
        {
            unsigned char *data = stbi_load(faces[x].c_str(), &width, &height, &nrComponents, 0);
            if (data)
            {
                GLenum format;
                if (nrComponents == 1)
                    format = GL_RED;
                else if (nrComponents == 3)
                    format = GL_RGB;
                else if (nrComponents == 4)
                    format = GL_RGBA;
                else
                    PRINT_ERROR("Unsupported image format!");
                glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + x, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
                stbi_image_free(data);
            }
            else
            {
                PRINT_ERROR("Cubemap texture failed to load at path" << faces[x]);
                stbi_image_free(data);
            }
        }
        // filter setting
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // minification filter
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // magnification filter
        // wrap setting
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // wrap on x axis
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // wrap on y axis
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE); // wrap on z axis
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    }

    void TextureCube::clearTexture()
    {
        glDeleteTextures(1, &m_id);
    }

    TextureDepth::TextureDepth(int width, int height)
        : m_width(width), m_height(height)
    {
        setupTexture("", "");
    }

    TextureDepth::~TextureDepth()
    {
        clearTexture();
    }

    void TextureDepth::bind(unsigned int slot)
    {
        glActiveTexture(GL_TEXTURE0 + slot);
        glBindTexture(GL_TEXTURE_2D, m_id);
    }

    void TextureDepth::unbind()
    {
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void TextureDepth::setupTexture(const std::string &path, const std::string &pFix)
    {
        // generate depth buffer.
        glGenTextures(1, &m_id);
        glBindTexture(GL_TEXTURE_2D, m_id);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
                     m_width, m_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        GLfloat borderColor[] = {1.0, 1.0, 1.0, 1.0};
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void TextureDepth::clearTexture()
    {
        glDeleteTextures(1, &m_id);
    }

    TextureColor::TextureColor(int width, int height, bool hdr)
        : m_width(width), m_height(height), m_hdr(hdr)
    {
        setupTexture("", "");
    }

    TextureColor::~TextureColor()
    {
        clearTexture();
    }

    void TextureColor::bind(unsigned int slot)
    {
        glActiveTexture(GL_TEXTURE0 + slot);
        glBindTexture(GL_TEXTURE_2D, m_id);
    }

    void TextureColor::unbind()
    {
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void TextureColor::setupTexture(const std::string &path, const std::string &pFix)
    {
        // generate depth buffer.
        glGenTextures(1, &m_id);
        glBindTexture(GL_TEXTURE_2D, m_id);
        if (!m_hdr)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_width, m_height,
                         0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        else
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, m_width, m_height,
                         0, GL_RGB, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        // here must be GL_NEAREST, otherwise there is a bug.
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void TextureColor::clearTexture()
    {
        glDeleteTextures(1, &m_id);
    }
}