#pragma once

#include <memory>
#include <string>
#include <glm/glm.hpp>

namespace Renderer
{

class Texture
{
public:
    typedef std::shared_ptr<Texture> ptr;
    Texture() = default;
    virtual ~Texture() = default;

    virtual void bind(unsigned int slot = 0) const = 0;
    virtual void unbind() const = 0;

    unsigned int getTextureId() const { return m_id; }

protected:
    std::string m_name;
    unsigned int m_id;

private:
    virtual void setupTexture(const std::string &path, const std::string &pFix) = 0;
    virtual void clearTexture() = 0;
};

class Texture2D : public Texture
{

public:
    typedef std::shared_ptr<Texture2D> ptr;

    Texture2D(unsigned char* images, int width, int height, int channels);
    Texture2D(const std::string& path, glm::vec4 bColor = glm::vec4(1.0f));
    ~Texture2D();

    virtual void bind(unsigned int slot);
    virtual void unBind();

private:
    virtual void setupTexture(const std::string &path, const std::string &pFix) override;
    virtual void clearTexture() override;

private:
    glm::vec4 m_borderColor;
    int m_width, m_height, m_channels;
};


class TextureCube: public Texture
{
public:
    TextureCube(const std::string &path, const std::string *posFix);
    ~TextureCube();

    virtual void bind(unsigned slot);
    virtual void unBind();
private:
    virtual void setupTexture(const std::string &path, const std::string &pFix);
    virtual void clearTexture();
};

class TextureDepth : public Texture
{
private:
    int m_width, m_height;

public:
    TextureDepth(int width, int height);
    ~TextureDepth();

    virtual void bind(unsigned int unit);
    virtual void unBind();

private:
    virtual void setupTexture(const std::string &path, const std::string &pFix);
    virtual void clearTexture();
};

class TextureColor : public Texture
{
private:
	bool m_hdr;
	int m_width, m_height;

public:
	TextureColor(int width, int height, bool hdr = false);
	~TextureColor();

	virtual void bind(unsigned int unit);
	virtual void unBind();

private:
	virtual void setupTexture(const std::string &path, const std::string &pFix);
	virtual void clearTexture();
};

}