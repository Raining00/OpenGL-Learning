#include "RenderApp/WindowApp.h"

class MyWindowApp : public Renderer::WindowApp
{
public:
    MyWindowApp(int width = 1920, int height = 1080, const std::string &title = "WindowApp", const std::string &cameraType = "tps")
        : WindowApp(width, height, title, cameraType)
    {
    }

    ~MyWindowApp() = default;

    virtual void Init() override
    {
        // you own init code here
    }

    virtual void Update() override
    {
        // you own update code here, may be physics simulation
    }

    virtual void Render() override
    {
        // you own render code here
        m_renderSystem->setClearColor(glm::vec4(m_BackColor, 1.0f));
        m_renderSystem->render();
        {
            // ImGUI 
            ImGui::Begin("This is a example of WindowApp");
            ImGui::ColorEdit3("Background Color", (float *)&m_BackColor);
            ImGui::End();
        }
    }

    virtual void Release() override
    {
    }

private:
    // you own member variables here such as VAO, VBO, EBO, Shader, Texture, etc.
    glm::vec3 m_BackColor{0.2f, 0.3f, 0.3f};
};

int main(int argc, char **argv)
{
    MyWindowApp app;
    app.Run();
    return 0;
}