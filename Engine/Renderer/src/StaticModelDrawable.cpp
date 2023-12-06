#include "StaticModelDrawable.h"

#include "MeshManager.h"
#include "TextureManager.h"
#include "ShaderManager.h"
#include "LightManager.h"

#include "ColorfulPrint.h"
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>

namespace Renderer
{

    StaticModelDrawable::StaticModelDrawable(unsigned int shaderIndex, const std::string &path)
    {
        m_shaderIndex = shaderIndex;
        loadModel(path);
    }

    void StaticModelDrawable::renderImp()
    {
        MeshManager::ptr meshManager = MeshManager::getSingleton();
        for (int x = 0; x < m_meshIndex.size(); x++)
        {
            this->bind(x);
			meshManager->drawMesh(m_meshIndex[x], m_instance, m_instanceNum);
            this->unbind(x);
		}
    }

    void StaticModelDrawable::render(Camera3D::ptr camera, Light::ptr sunLight, Camera3D::ptr lightCamera, Shader::ptr shader)
    {
        if (!m_visible)
            return;

        glStencilMask(0x00);
        if (m_stencil)
        {
            glEnable(GL_STENCIL_TEST);
            glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
            glStencilFunc(GL_ALWAYS, 1, 0xFF);
            glStencilMask(0xFF);
        }

        if (shader == nullptr)
        {
            shader = ShaderManager::getInstance()->getShader(m_shaderIndex);
            shader->use();
        }
        if (sunLight)
        {
            sunLight->setLightUniforms(shader, camera, "sunLight");
        }
        shader->setInt("material.diffuse", 0);
        shader->setInt("material.specular", 1);
        shader->setInt("material.normal", 2);
        shader->setInt("material.height", 3);
        shader->setInt("material.reflection", 4);
        if (lightCamera != nullptr)
            shader->setMat4("lightSpaceMatrix", lightCamera->getProjectionMatrix() * lightCamera->getViewMatrix());
        else
            shader->setMat4("lightSpaceMatrix", glm::mat4(1.0f));
        // object matrix
        shader->setBool("instance", false);
        shader->setBool("receiveShadow", m_receiveShadow);
        shader->setMat4("modelMatrix", m_transformation.getWorldMatrix());
        //shader->setMat4("viewMatrix", camera->getViewMatrix());
        //shader->setMat4("projectMatrix", camera->getProjectionMatrix());
        shader->setMat3("normalMatrix", m_transformation.getNormalMatrix());
        this->renderImp();

        // use stencil test to draw outline
        if (m_stencil)
		{
            // for example: glStencilFunc(GL_NOTEQUAL, 1, 0xFF);
            // glStencilMask(0x00);
            glStencilFunc(m_stencilOp.func, m_stencilOp.ref, m_stencilOp.funcMask);
            glStencilMask(m_stencilOp.stencilMask);
			glDisable(GL_DEPTH_TEST);
			//Shader::ptr shader = ShaderManager::getInstance()->getShader("stencil");
            shader = ShaderManager::getInstance()->getShader(m_stencilShaderIndex);
			shader->use();
            float scale = 1.1f;
            glm::mat4 modelMatrix(1.0f);
            modelMatrix = glm::translate(modelMatrix, m_transformation.translation());
            modelMatrix = glm::scale(modelMatrix, glm::vec3(scale, scale, scale));

			shader->setMat4("modelMatrix", modelMatrix);
			shader->setMat4("viewMatrix", camera->getViewMatrix());
			shader->setMat4("projectMatrix", camera->getProjectionMatrix());
            shader->setMat3("normalMatrix", m_transformation.getNormalMatrix());
			this->renderImp();
            // reset and clear stencil buffer. make sure this will not affect other objects.
			glStencilMask(0xFF);
			glStencilFunc(GL_ALWAYS, 1, 0xFF);
			glEnable(GL_DEPTH_TEST);
            glClear(GL_STENCIL_BUFFER_BIT);
		}

        ShaderManager::getInstance()->unbindShader();
    }

    void StaticModelDrawable::renderDepth(Shader::ptr shader, Camera3D::ptr lightCamera)
    {
        if (!m_visible || !m_produceShadow)
            return;
        shader->use();
        shader->setBool("instance", false);
        shader->setMat4("lightSpaceMatrix",
                        lightCamera->getProjectionMatrix() * lightCamera->getViewMatrix());
        shader->setMat4("modelMatrix", m_transformation.getWorldMatrix());
        this->renderImp();
        ShaderManager::getSingleton()->unbindShader();
    }

    void StaticModelDrawable::unbind(int x)
    {
        TextureManager::ptr textureManager = TextureManager::getSingleton();
        if (m_textureMapList[x].find("diffuse") != m_textureMapList[x].end())
            textureManager->unbindTexture(m_textureMapList[x]["diffuse"]);
        if (m_textureMapList[x].find("specular") != m_textureMapList[x].end())
            textureManager->unbindTexture(m_textureMapList[x]["specular"]);
        if (m_textureMapList[x].find("normal") != m_textureMapList[x].end())
            textureManager->unbindTexture(m_textureMapList[x]["normal"]);
        if (m_textureMapList[x].find("height") != m_textureMapList[x].end())
            textureManager->unbindTexture(m_textureMapList[x]["height"]);
        if (textureManager->getTexture("skybox") != nullptr)
            textureManager->unbindTexture(textureManager->getTextureIndex("skybox"));
    }

    void StaticModelDrawable::bind(int x)
    {
        TextureManager::ptr textureManager = TextureManager::getSingleton();
        if (m_textureMapList[x].find("diffuse") != m_textureMapList[x].end())
            textureManager->bindTexture(m_textureMapList[x]["diffuse"], 0);
        if (m_textureMapList[x].find("specular") != m_textureMapList[x].end())
            textureManager->bindTexture(m_textureMapList[x]["specular"], 1);
        if (m_textureMapList[x].find("normal") != m_textureMapList[x].end())
            textureManager->bindTexture(m_textureMapList[x]["normal"], 2);
        if (m_textureMapList[x].find("height") != m_textureMapList[x].end())
            textureManager->bindTexture(m_textureMapList[x]["height"], 3);
        if (textureManager->getTexture("skybox") != nullptr)
            textureManager->bindTexture(textureManager->getTextureIndex("skybox"), 4);
    }


    void StaticModelDrawable::loadModel(const std::string& path)
    {
        //load the model file
        m_min = glm::vec3(FLT_MAX);
        m_max = glm::vec3(-FLT_MAX);
        Assimp::Importer importer;
        const aiScene *scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
        {
            PRINT_ERROR("ERROR::ASSIMP::" << std::string(importer.GetErrorString()));
            return;
        }
        m_directory = path.substr(0, path.find_last_of('/'));
        processNode(scene->mRootNode, scene);
    }

    void StaticModelDrawable::processNode(aiNode* node, const aiScene* scene)
    {
        // process all the node's meshes (if any)
        for (unsigned int i = 0; i < node->mNumMeshes; ++i)
        {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            unsigned int meshIndex, texIndex = 1000000;
            processMesh(mesh, scene, meshIndex, texIndex);
            this->addMesh(meshIndex);
            if(texIndex != 1000000)
                this->addTexture(texIndex);
        }
        // then do the same for each of its children
        for (unsigned int i = 0; i < node->mNumChildren; ++i)
        {
            processNode(node->mChildren[i], scene);
        }
    }

    void StaticModelDrawable::processMesh(aiMesh* mesh, const aiScene* scene, unsigned int& meshIndex, unsigned int& texIndex)
    {
        // process mesh
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;
        for(unsigned int x = 0; x < mesh->mNumVertices; x++)
        {
            Vertex vertex;
            // process vertex positions, normals and texture coordinates
            // position
            vertex.position = glm::vec3(mesh->mVertices[x].x, mesh->mVertices[x].y, mesh->mVertices[x].z);
            // normal
            vertex.normal = glm::vec3(mesh->mNormals[x].x, mesh->mNormals[x].y, mesh->mNormals[x].z);
            // texture coordinates
            if(mesh->mTextureCoords[0])
            {
                vertex.texCoords = glm::vec2(mesh->mTextureCoords[0][x].x, mesh->mTextureCoords[0][x].y);
            }
            else
            {
                vertex.texCoords = glm::vec2(0.0f, 0.0f);
            }
            vertex.color = vertex.normal;
            vertices.push_back(vertex);
            // bounding box
            if (mesh->mVertices[x].x < m_min.x)
                m_min.x = mesh->mVertices[x].x;
            if (mesh->mVertices[x].y < m_min.y)
                m_min.y = mesh->mVertices[x].y;
            if (mesh->mVertices[x].z < m_min.z)
                m_min.z = mesh->mVertices[x].z;
            if (mesh->mVertices[x].x > m_max.x)
                m_max.x = mesh->mVertices[x].x;
            if (mesh->mVertices[x].y > m_max.y)
                m_max.y = mesh->mVertices[x].y;
            if (mesh->mVertices[x].z > m_max.z)
                m_max.z = mesh->mVertices[x].z;
        }

        for (unsigned int x = 0; x<mesh->mNumFaces; x++)
        {
            aiFace face = mesh->mFaces[x];
            for (unsigned int y = 0; y<face.mNumIndices; y++)
            {
                indices.push_back(face.mIndices[y]);
            }
        }

        Mesh* target = new Mesh(vertices, indices);
        meshIndex = MeshManager::getInstance()->loadMesh(target);

        // process material
        TextureManager::ptr textureManager = TextureManager::getInstance();
        if(mesh->mMaterialIndex >= 0)
        {
            aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
            aiString nameStr;
            unsigned int count = material->GetTextureCount(aiTextureType_DIFFUSE);
            std::map<std::string, unsigned int> tmp;
            if (count > 0)
            {
                material->GetTexture(aiTextureType_DIFFUSE, 0, &nameStr);
                std::string name(nameStr.C_Str());
                if (name != "")
                {
                    texIndex = textureManager->loadTexture2D(name, m_directory + "/" + name);
                    tmp.insert(std::make_pair("diffuse", texIndex));
                }
            }
            count = material->GetTextureCount(aiTextureType_SPECULAR);
            if (count > 0) {
                material->GetTexture(aiTextureType_SPECULAR, 0, &nameStr);
                std::string name = std::string(nameStr.C_Str());
                if (name != "")
                {
                    texIndex = textureManager->loadTexture2D(name, m_directory + "/" + name);
                    tmp.insert(std::make_pair("specular", texIndex));
                }
            }
            count = material->GetTextureCount(aiTextureType_HEIGHT);
            if (count > 0)
            {
				material->GetTexture(aiTextureType_HEIGHT, 0, &nameStr);
				std::string name = std::string(nameStr.C_Str());
                if (name != "")
                {
					texIndex = textureManager->loadTexture2D(name, m_directory + "/" + name);
					tmp.insert(std::make_pair("normal", texIndex));
				}
			}
            count = material->GetTextureCount(aiTextureType_AMBIENT);
            if (count > 0)
            {
                material->GetTexture(aiTextureType_AMBIENT, 0, &nameStr);
				std::string name = std::string(nameStr.C_Str());
                if (name != "")
                {
					texIndex = textureManager->loadTexture2D(name, m_directory + "/" + name);
					tmp.insert(std::make_pair("height", texIndex));
				}
            }
            m_textureMapList.push_back(tmp);
        }

    }

} // namespace Renderer
