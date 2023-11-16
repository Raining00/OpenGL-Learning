#include "RenderDevice.h"
#include "ColorfulPrint.h"

namespace Renderer
{

glm::vec2 RenderDevice::m_cursorPos;
glm::vec2 RenderDevice::m_deltaCurPos;
float RenderDevice::m_deltaTime = 0.0f, RenderDevice::m_lastFrame = 0.0f;
bool RenderDevice::m_keyPressed[1024] = {0};
bool RenderDevice::m_buttonPressed[GLFW_MOUSE_BUTTON_LAST] = {0};
template<> std::shared_ptr<RenderDevice> Singleton<RenderDevice>::_instance = nullptr;

std::shared_ptr<RenderDevice> RenderDevice::getInstance()
{
    if (_instance == nullptr)
    {
        _instance = std::shared_ptr<RenderDevice>(new RenderDevice());
    }
    return _instance;
}

bool RenderDevice::initialize(const std::string& title, const int& width, const int& height, bool debug)
{
    m_width = width;
    m_height = height;
    m_lastFrame = 0.0f;
    m_deltaTime = 0.0f;
    m_debugMode = debug;
    m_cursorPos.x = m_width / 2.0f;
    m_cursorPos.y = m_height / 2.0f;

    // context initialization
    if (!glfwInit())
    {
        PRINT_ERROR("Failed to initialize GLFW");
        return false;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    //DEBUG
    if(m_debugMode)
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
    else
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_FALSE);
    m_windowHandler = glfwCreateWindow(m_width, m_height, title.c_str(), nullptr, nullptr);
    if(m_windowHandler == nullptr)
    {
        PRINT_ERROR("Failed to create GLFW window");
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(m_windowHandler);
    glfwSetInputMode(m_windowHandler, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // load OpenGL context
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        PRINT_ERROR("Failed to initialize GLAD");
        return false;
    }

    // initialize debug context
    if(m_debugMode)
        initializeDebugContex();
    
    //check error
    GLenum err = glGetError();
    if(err != GL_NO_ERROR)
    {
        PRINT_ERROR("OpenGL error: " << err);
        return false;
    }

    // imgui context
    m_ImGuiContext = std::make_shared<ImGui::ImGuiOpenGLContext>(m_windowHandler);

    glfwSetCursorPosCallback(m_windowHandler, cursorPositionCallback);
    glfwSetMouseButtonCallback(m_windowHandler, mouseButtonCallback);
    glfwSetKeyCallback(m_windowHandler, keyCallback);
    glfwSetFramebufferSizeCallback(m_windowHandler, framebufferSizeCallback);
    glfwSetScrollCallback(m_windowHandler, scrollCallback);

    glEnable(GL_MULTISAMPLE);

    return true;
}

bool RenderDevice::run()
{
    if(m_windowHandler == nullptr)
    {
        PRINT_ERROR("Failed to create GLFW window");
        glfwTerminate();
        return false;
    }
    return !glfwWindowShouldClose(m_windowHandler);
}

void RenderDevice::beginFrame()
{
    processInput();
    m_ImGuiContext->BeginFrame();

}

void RenderDevice::endFrame()
{
    // fps display
    {
        ImGui::Begin("Logger: ");
        ImGui::Text("Frame Rate");
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
	                    1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();
    }

    //delete time calculation
    float currentFrame = glfwGetTime();
    m_deltaTime = currentFrame - m_lastFrame;
    m_lastFrame = currentFrame;

    m_ImGuiContext->EndFrame();
    glfwSwapBuffers(m_windowHandler);
    glfwPollEvents();
}

bool RenderDevice::shutdown()
{
    m_ImGuiContext->Shutdown();
    glfwTerminate();
    return true;
}

void RenderDevice::initializeDebugContex()
{
    GLint flags;
    glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
    if (flags & GL_CONTEXT_FLAG_DEBUG_BIT)
    {
        std::cout << "OpenGL Debug Mode Launched.\n";
        glEnable(GL_DEBUG_OUTPUT);
        // makes sure errors are displayed synchronously.
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback((GLDEBUGPROC)glDebugOutput, nullptr);
        glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
    }
}

void RenderDevice::glDebugOutput(GLenum source, GLenum type, GLuint id, GLenum severity,
	GLsizei length, const GLchar * message, const void * userParam)
{
	// ignore non-significant error/warning codes
	if (id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

	std::cout << "---------------" << std::endl;
	std::cout << "Debug message (" << id << "): " << message << std::endl;

	switch (source)
	{
	case GL_DEBUG_SOURCE_API:             std::cout << "Source: API"; break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cout << "Source: Window System"; break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cout << "Source: Shader Compiler"; break;
	case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cout << "Source: Third Party"; break;
	case GL_DEBUG_SOURCE_APPLICATION:     std::cout << "Source: Application"; break;
	case GL_DEBUG_SOURCE_OTHER:           std::cout << "Source: Other"; break;
	} 
	std::cout << std::endl;

	switch (type)
	{
	case GL_DEBUG_TYPE_ERROR:               std::cout << "Type: Error"; break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::cout << "Type: Deprecated Behaviour"; break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::cout << "Type: Undefined Behaviour"; break;
	case GL_DEBUG_TYPE_PORTABILITY:         std::cout << "Type: Portability"; break;
	case GL_DEBUG_TYPE_PERFORMANCE:         std::cout << "Type: Performance"; break;
	case GL_DEBUG_TYPE_MARKER:              std::cout << "Type: Marker"; break;
	case GL_DEBUG_TYPE_PUSH_GROUP:          std::cout << "Type: Push Group"; break;
	case GL_DEBUG_TYPE_POP_GROUP:           std::cout << "Type: Pop Group"; break;
	case GL_DEBUG_TYPE_OTHER:               std::cout << "Type: Other"; break;
	}
	std::cout << std::endl;

	switch (severity)
	{
	case GL_DEBUG_SEVERITY_HIGH:         std::cout << "Severity: high"; break;
	case GL_DEBUG_SEVERITY_MEDIUM:       std::cout << "Severity: medium"; break;
	case GL_DEBUG_SEVERITY_LOW:          std::cout << "Severity: low"; break;
	case GL_DEBUG_SEVERITY_NOTIFICATION: std::cout << "Severity: notification"; break;
	}
	std::cout << std::endl;
	std::cout << std::endl;
}


void RenderDevice::cursorPositionCallback(GLFWwindow * window, double xpos, double ypos)
{
    static bool first = true;
    if ((!m_buttonPressed[GLFW_MOUSE_BUTTON_LEFT] && !m_buttonPressed[GLFW_MOUSE_BUTTON_RIGHT]))
    {
        first = true;
        return;
    }

    // delta cursor position.
    glm::vec2 lastPos = m_cursorPos;
    if (first)
    {
        lastPos = glm::vec2(xpos, ypos);
        first = false;
    }
    m_deltaCurPos.x = xpos - lastPos.x;
    m_deltaCurPos.y = lastPos.y - ypos;
    m_cursorPos = glm::vec2(xpos, ypos);
}

void RenderDevice::mouseButtonCallback(GLFWwindow * window, int button, int action, int mods)
{
    if (action == GLFW_PRESS)
        m_buttonPressed[button] = true;
    else if (action == GLFW_RELEASE)
        m_buttonPressed[button] = false;
}

void RenderDevice::keyCallback(GLFWwindow * window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
        m_keyPressed[key] = true;
    else if (action == GLFW_RELEASE)
        m_keyPressed[key] = false;
}

void RenderDevice::framebufferSizeCallback(GLFWwindow * window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void RenderDevice::scrollCallback(GLFWwindow * window, double xoffset, double yoffset)
{
    //TODO: add scroll callback
}

void RenderDevice::processInput()
{
    // shutdown.
    if (glfwGetKey(m_windowHandler, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(m_windowHandler, true);

    // TODO: add camera control.
    // std::shared_ptr<RenderDevice> device = getSingleton();

    // // key pressed action.
    // if (m_keyPressed[GLFW_KEY_W])
    //     device->getRenderSystem()->getCamera()->onKeyPress(m_deltaTime, 'W');
    // if (m_keyPressed[GLFW_KEY_A])
    //     device->getRenderSystem()->getCamera()->onKeyPress(m_deltaTime, 'A');
    // if (m_keyPressed[GLFW_KEY_S])
    //     device->getRenderSystem()->getCamera()->onKeyPress(m_deltaTime, 'S');
    // if (m_keyPressed[GLFW_KEY_D])
    //     device->getRenderSystem()->getCamera()->onKeyPress(m_deltaTime, 'D');
    // if (m_keyPressed[GLFW_KEY_Q])
    //     device->getRenderSystem()->getCamera()->onKeyPress(m_deltaTime, 'Q');
    // if (m_keyPressed[GLFW_KEY_E])
    //     device->getRenderSystem()->getCamera()->onKeyPress(m_deltaTime, 'E');

    // // mouse action.
    // if (m_buttonPressed[GLFW_MOUSE_BUTTON_LEFT])
    //     device->getRenderSystem()->getCamera()->onMouseMove(m_deltaCurPos.x, m_deltaCurPos.y, "LEFT");
    // else if (m_buttonPressed[GLFW_MOUSE_BUTTON_RIGHT])
    //     device->getRenderSystem()->getCamera()->onMouseMove(m_deltaCurPos.x, m_deltaCurPos.y, "RIGHT");
    // m_deltaCurPos = glm::vec2(0.0f);
}

} // namespace Render