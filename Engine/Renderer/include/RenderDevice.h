#pragma once

#include <string>
#include "ImGuiOpenGLContext.h"
#include "glm/glm.hpp"
#include "Singleton.h"

namespace Renderer
{

class RenderDevice: public Singleton<RenderDevice>
{
public:
    static glm::vec2 m_cursorPos;
    static glm::vec2 m_deltaCurPos;
    static bool m_keyPressed[1024];
    static float m_deltaTime, m_lastFrame;
    static bool m_buttonPressed[GLFW_MOUSE_BUTTON_LAST];

    RenderDevice() = default;
    ~RenderDevice() = default;

    static std::shared_ptr<RenderDevice> getInstance();

    bool initialize(const std::string& title, const int& width, const int& height, bool debug = false);
    bool run();
    void beginFrame();
    void endFrame();
    bool shutdown();

    int getWindowWidth() const { return m_width; }
    int getWindowHeight() const { return m_height; }

protected:
    void initializeDebugContex();
    static void APIENTRY glDebugOutput(GLenum source, GLenum type, GLuint id, GLenum severity, 
            GLsizei length, const GLchar* message, const void* userParam);
    
    //callback functions
    static void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    void processInput();

private:
    GLFWwindow* m_Window;
    bool m_debugMode;
    int m_width, m_height;
    GLFWwindow* m_windowHandler;
    std::shared_ptr<ImGui::ImGuiOpenGLContext> m_ImGuiContext;
};



}