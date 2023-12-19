#pragma once

#include <memory>
#include <string>
#include <glm/glm.hpp>

namespace Renderer
{
    enum class TextureType{
        AMBIENT,
        DIFFUSE,
        SPECULAR,
        NORMAL, 
        HEIGHT,
        REFLECT,
        COLOR,
        DEPTH,
        DEPTH_CUBE
    };

    class Texture
    {
    public:
        typedef std::shared_ptr<Texture> ptr;
        Texture() = default;
        virtual ~Texture() = default;

        virtual void bind(unsigned int slot) = 0;
        virtual void unbind() = 0;

        unsigned int getTextureId() const { return m_id; }
        TextureType getTextureType() const { return m_textureType; }
    protected:
        std::string m_name;
        unsigned int m_id;
        TextureType m_textureType;

    private:
        virtual void setupTexture(const std::string &path, const std::string &pFix) = 0;
        virtual void clearTexture() = 0;
    };

    /**
     * @brief This is a 2D texture class. It is used to create and manage standard 2D textures. Texture2D has two constructors:
     * one that receives the loaded image data (pixel array, width, height, etc.) and another that loads the image from a file path.
     * It supports different channel formats such as red, RGB, RGBA, and can set different texture filtering and wrapping modes.
     * This type of texture is very common in 3D graphics and is used for various surface maps such as diffuse maps, specular maps, etc.
     */
    class Texture2D : public Texture
    {

    public:
        typedef std::shared_ptr<Texture2D> ptr;

        Texture2D(unsigned char *images, int width, int height, int channels, const TextureType& textureType = TextureType::AMBIENT);
        Texture2D(const std::string &path, glm::vec4 bColor = glm::vec4(1.0f), const TextureType& textureType = TextureType::AMBIENT);
        ~Texture2D();

        virtual void bind(unsigned int slot);
        virtual void unbind();

    private:
        virtual void setupTexture(const std::string& path, const std::string& pFix);
        virtual void clearTexture();

    private:
        glm::vec4 m_borderColor;
        int m_width, m_height, m_channels;
    };

    /**
     * @brief This is a cube texture class used to create and manage cubemaps.
     * Cube textures are mainly used to create environment maps such as skyboxes or reflections.
     * The TextureCube class loads images from six different image files (one for each face) to create a cube texture.
     * These textures are often used to achieve reflection and refraction effects, or to simulate the background of an open space.
     *
     */
    class TextureCube : public Texture
    {
    public:
        TextureCube(const std::string &path, const std::string& posFix);
        ~TextureCube();

        virtual void bind(unsigned slot);
        virtual void unbind();

    private:
        virtual void setupTexture(const std::string& path, const std::string& pFix);
        virtual void clearTexture();
    };

    /**
     * @brief This is a depth texture class used to create and manage depth buffers. 
     * It is not loaded from an image file but generated directly on the GPU. 
     * Depth texture is mainly used for depth testing, which can store distance information from the camera viewpoint to the object surface. 
     * This is useful in the implementation of shadow mapping and depth effects such as depth of field
     * 
     */
    class TextureDepth : public Texture
    {
    private:
        int m_width, m_height;

    public:
        TextureDepth(int width, int height);
        ~TextureDepth();

        virtual void bind(unsigned int unit);
        virtual void unbind();

    private:
        virtual void setupTexture(const std::string& path, const std::string& pFix);
        virtual void clearTexture();
    };

    class TextureDepthCube : public Texture
    {
    private:
        	int m_width, m_height;
    public:
        TextureDepthCube(int width, int height);
		~TextureDepthCube();

		virtual void bind(unsigned int unit);
		virtual void unbind();
    private:
        virtual void setupTexture(const std::string& path, const std::string& pFix);
		virtual void clearTexture();
    };

    /**
     * @brief This is a color texture class for creating color buffers with or without high dynamic range (HDR).
     * Like TextureDepth, it is generated directly on the GPU and is used to store color information. 
     * This type of texture can be used in off-screen rendering (Off-Screen Rendering), 
     * such as as a render target (Render Target) when implementing post-processing effects.
     */
    class TextureColor : public Texture
    {
    private:
        bool m_hdr;
        int m_width, m_height;

    public:
        TextureColor(int width, int height, bool hdr = false);
        ~TextureColor();

        virtual void bind(unsigned int unit);
        virtual void unbind();

    private:
        virtual void setupTexture(const std::string& path, const std::string& pFix);
        virtual void clearTexture();
    };

    class TextureStencil :public Texture
    {

    public:
        TextureStencil(int width, int height);
		~TextureStencil();

		virtual void bind(unsigned int unit);
		virtual void unbind();

    private:
        virtual void setupTexture(const std::string& path, const std::string& pFix);
        virtual void clearTexture();

    private:
        int m_width, m_height;
    };


    class TextureDepthStencil :public Texture
    {
    public:
        TextureDepthStencil(int width, int height);
		~TextureDepthStencil();

		virtual void bind(unsigned int unit);
		virtual void unbind();

    private:
        virtual void setupTexture(const std::string& path, const std::string& pFix);
		virtual void clearTexture();

    private:
        int m_width, m_height;
    };
}