#include "RenderApp/RenderDevice.h"
#include "Manager/TextureManager.h"
#include "Manager/ShaderManager.h"
#include "ColorfulPrint.h"
#include "Config.h"

#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

int main()
{
    PRINT_INFO("Start Transformations");
    auto window = Renderer::RenderDevice::getInstance();
    window->initialize("Transformations", 1920, 1080);
    Renderer::TextureManager::ptr TexMgr = Renderer::TextureManager::getInstance();
    Renderer::ShaderManager::ptr ShaderMgr = Renderer::ShaderManager::getInstance();

    float vertices[] = {
        //      ---- 位置 ----       ---- 颜色 ----     - 纹理坐标 -
        0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,   
        0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,  
        -0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 
        -0.5f, 0.5f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f   
    };

    unsigned int indices[] = {
        0, 1, 3, // 第一个三角形
        1, 2, 3  // 第二个三角形
    };

    GLuint shader1 = ShaderMgr->loadShader("texture", SHADER_PATH "/Transformations/transformations.vs", SHADER_PATH "/Transformations/transformations.fs");

    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // create texture
    GLuint tex1 = TexMgr->loadTexture2D("tomorin", ASSETS_PATH "/texture/109447235_p0.jpg");
    GLuint tex2 = TexMgr->loadTexture2D("pic2", ASSETS_PATH "/texture/93447255_p0.png");
    glm::vec3 backgroundColor(0.2f, 0.3f, 0.3f);
    glm::vec3 scale(1.0f, 1.0f, 1.0f);
    float mixValue = 0.2f;
    ShaderMgr->bindShader(shader1);
    ShaderMgr->getShader(shader1)->setInt("texture1", 0);
    ShaderMgr->getShader(shader1)->setInt("texture2", 1);
    ShaderMgr->unbindShader();

    while(window->run())
    {
        window->beginFrame();
        
        glClearColor(backgroundColor.x, backgroundColor.y, backgroundColor.z, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // bind texture
        ShaderMgr->bindShader(shader1);
        glm::mat4 trans(1.0);
        trans = glm::translate(trans, glm::vec3(0.5f, -0.5f, 0.0f));
        trans = glm::rotate(trans, (float)glfwGetTime(), glm::vec3(0.0f, 0.0f, 1.0f));
        ShaderMgr->getShader(shader1)->setMat4("transform", glm::value_ptr(trans));

        TexMgr->bindTexture(tex1, 0);
        TexMgr->bindTexture(tex2, 1);
        ShaderMgr->getShader(shader1)->setFloat("mixValue", mixValue);
        // draw triangle
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        ShaderMgr->unbindShader();
        
        // imgui
        {
            ImGui::Begin("Transformations");
            ImGui::Text("Transformation Example");
            ImGui::ColorEdit3("backgroundColor", (float*)&backgroundColor);
            ImGui::SliderFloat3("scale", (float*)&scale, 0.0f, 2.0f);
            ImGui::SliderFloat("mixValue", &mixValue, 0.0f, 1.0f);
            ImGui::End();
        }
        window->endFrame();
    }
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteVertexArrays(1, &VAO);
    window->shutdown();

    return 0;
}