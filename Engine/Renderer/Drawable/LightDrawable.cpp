#include "Drawable/LightDrawable.h"
#include "Manager/LightManager.h"
#include "Manager/ShaderManager.h"
#include "Manager/MeshManager.h"
#include "include/Config.h"

Renderer::LightDrawable::LightDrawable(const std::string& name) :m_lightName(name)
{
	m_lightIndex = LightManager::getInstance()->CreatePointLight(m_lightName, getTransformation()->translation(), m_amibent, m_diffuse, m_specular, m_constant, m_linear, m_quadratic);
	m_shaderIndex = ShaderManager::getInstance()->loadShader("LightShader", SHADER_PATH"/LightDrawable/SimpleLightDrawable.vs", SHADER_PATH"/LightDrawable/SimpleLightDrawable.fs");

}

Renderer::LightDrawable::LightDrawable(const std::string& name, const glm::vec3& ambient, const glm::vec3& diffuse, const glm::vec3& specular, const float& constant, const float& linear, const float& quadratic, const glm::vec3& position)
	:m_lightName(name), m_amibent(ambient), m_diffuse(diffuse), m_specular(specular), m_constant(constant), m_linear(linear), m_quadratic(quadratic)
{
    getTransformation()->setTranslation(position);
	m_lightIndex = LightManager::getInstance()->CreatePointLight(m_lightName, position, m_amibent, m_diffuse, m_specular, m_constant, m_linear, m_quadratic);
	m_shaderIndex = ShaderManager::getInstance()->loadShader("LightShader", SHADER_PATH"/LightDrawable/SimpleLightDrawable.vs", SHADER_PATH"/LightDrawable/SimpleLightDrawable.fs");
}

void Renderer::LightDrawable::render(Camera3D::ptr camera, Camera3D::ptr lightCamera, Shader::ptr shader)
{
    if (!m_visible)
        return;
    // disable stencil test.
    glStencilMask(0x00);
    if (m_stencil)
    {
        glEnable(GL_STENCIL_TEST);
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
        glStencilFunc(GL_ALWAYS, 1, 0xFF);
        glStencilMask(0xFF);
    }
    if (shader == nullptr)
        shader = ShaderManager::getSingleton()->getShader(m_shaderIndex);
    shader->use();
    //Light::ptr sunLight = LightManager::getInstance()->getLight("sunLight");
    //if (sunLight)
    //    sunLight->setLightUniforms(shader, camera, "sunLight");
    LightManager::getInstance()->setLight(shader, camera);
    // texture
    shader->setInt("material.diffuse", 0);
    shader->setInt("material.specular", 0);

    // object matrix.
    shader->setBool("instance", m_instance);
    shader->setBool("receiveShadow", m_receiveShadow);
    shader->setMat4("modelMatrix", m_transformation.getWorldMatrix());
    shader->setMat3("normalMatrix", m_transformation.getNormalMatrix());
    shader->setVec3("lightColor", m_lightColor);
    this->renderImp();
    ShaderManager::getSingleton()->unbindShader();
}

void Renderer::LightDrawable::renderImp()
{
    MeshManager::ptr meshManager = MeshManager::getSingleton();
    TextureManager::ptr textureManager = TextureManager::getSingleton();
    for (int x = 0; x < m_meshIndex.size(); x++)
    {
        for (int i = 0; i < m_texIndex.size(); i++)
            textureManager->bindTexture(m_texIndex[x], i);
        meshManager->drawMesh(m_meshIndex[x], m_instance, m_instanceNum);
    }
    if (m_texIndex.size() > 0)
        textureManager->unbindTexture(m_texIndex[0]);
}

void Renderer::LightDrawable::updateLightSetting()
{
    PointLight* pointLight = reinterpret_cast<PointLight*>(LightManager::getInstance()->getLight(m_lightIndex).get());
    pointLight->setLightAttenuation(m_constant, m_linear, m_quadratic);
    pointLight->setLightColor(m_amibent, m_diffuse, m_specular);
    pointLight->setLightPosition(getTransformation()->translation());
}

