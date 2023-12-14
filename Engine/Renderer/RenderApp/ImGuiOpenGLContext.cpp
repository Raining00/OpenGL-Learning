#include "ImGuiOpenGLContext.h"
#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"

namespace ImGui
{
	ImGuiOpenGLContext* imguiOpenGLContext;

	ImGuiOpenGLContext::ImGuiOpenGLContext(GLFWwindow* window)
	{
		imguiOpenGLContext = this;
		this->Init(window);
		uFontTexture = 0;
	}

	ImGuiOpenGLContext::~ImGuiOpenGLContext()
	{
		this->Shutdown();
	}

	void ImGuiOpenGLContext::Init(GLFWwindow* window)
	{
		pWindow = window;
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		m_shutdown = false;
		ImGuiIO& io = ImGui::GetIO(); (void)io;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

		io.KeyMap[ImGuiKey_Tab] = GLFW_KEY_TAB;
		io.KeyMap[ImGuiKey_LeftArrow] = GLFW_KEY_LEFT;
		io.KeyMap[ImGuiKey_RightArrow] = GLFW_KEY_RIGHT;
		io.KeyMap[ImGuiKey_UpArrow] = GLFW_KEY_UP;
		io.KeyMap[ImGuiKey_DownArrow] = GLFW_KEY_DOWN;
		io.KeyMap[ImGuiKey_PageUp] = GLFW_KEY_PAGE_UP;
		io.KeyMap[ImGuiKey_PageDown] = GLFW_KEY_PAGE_DOWN;
		io.KeyMap[ImGuiKey_Home] = GLFW_KEY_HOME;
		io.KeyMap[ImGuiKey_End] = GLFW_KEY_END;
		io.KeyMap[ImGuiKey_Delete] = GLFW_KEY_DELETE;
		io.KeyMap[ImGuiKey_Backspace] = GLFW_KEY_BACKSPACE;
		io.KeyMap[ImGuiKey_Enter] = GLFW_KEY_ENTER;
		io.KeyMap[ImGuiKey_Escape] = GLFW_KEY_ESCAPE;
		io.KeyMap[ImGuiKey_A] = GLFW_KEY_A;
		io.KeyMap[ImGuiKey_C] = GLFW_KEY_C;
		io.KeyMap[ImGuiKey_V] = GLFW_KEY_V;
		io.KeyMap[ImGuiKey_X] = GLFW_KEY_X;
		io.KeyMap[ImGuiKey_Y] = GLFW_KEY_Y;
		io.KeyMap[ImGuiKey_Z] = GLFW_KEY_Z;

        // Setup Platform/Renderer backends
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 420");

		/* 设置剪切板回调函数 */
		io.SetClipboardTextFn = ImGuiOpenGLContext::SetClipboardText;
		io.GetClipboardTextFn = ImGuiOpenGLContext::GetClipboardText;
		io.ClipboardUserData = pWindow;

		/* 设置消息回调函数 */
		glfwSetKeyCallback(pWindow, ImGuiOpenGLContext::KeyCallback);
		glfwSetCharCallback(pWindow, ImGuiOpenGLContext::CharCallback);
		glfwSetScrollCallback(pWindow, ImGuiOpenGLContext::ScrollCallback);
		glfwSetMouseButtonCallback(pWindow, ImGuiOpenGLContext::MouseButtonCallback);
	}

	void ImGuiOpenGLContext::Shutdown()
	{
		if(!m_shutdown)
		{
			ImGui_ImplOpenGL3_Shutdown();
			ImGui_ImplGlfw_Shutdown();
			ImGui::DestroyContext();
			m_shutdown = true;
		}
	}

	void ImGuiOpenGLContext::BeginFrame()
	{
		ImGuiIO& io = ImGui::GetIO();
		/* 显示区域大小（不一定和窗口大小相等） */
		static int w, h, display_w, display_h;
		glfwGetWindowSize(pWindow, &w, &h);
		glfwGetFramebufferSize(pWindow, &display_w, &display_h);
        
		io.DisplaySize = ImVec2(float(w), float(h));
		io.DisplayFramebufferScale = ImVec2(w > 0 ? ((float)display_w / w) : 0, h > 0 ? ((float)display_h / h) : 0);

		/* 帧间隔时间 */
		double current_time = glfwGetTime();
		io.DeltaTime = fTime >= 0 ? (float)(current_time - fTime) : float(1.0f / 60.0f);
		fTime = current_time;

		/* 光标坐标 */
		if (glfwGetWindowAttrib(pWindow, GLFW_FOCUSED)) {
			static double mousex, mousey;
			glfwGetCursorPos(pWindow, &mousex, &mousey);

			io.MousePos = ImVec2((float)mousex, (float)mousey);
		}
		else {
			io.MousePos = ImVec2(-1, -1);
		}

		/* 鼠标按键 */
		for (int i = 0; i < 3; i++) {
			io.MouseDown[i] = mousePressed[i] || glfwGetMouseButton(pWindow, i) != 0;
			mousePressed[i] = false;
		}

		io.MouseWheel = fMouseWheel;
		fMouseWheel = 0.0f;

		glfwSetInputMode(pWindow, GLFW_CURSOR, io.MouseDrawCursor ? GLFW_CURSOR_HIDDEN : GLFW_CURSOR_NORMAL);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
	}

	void ImGuiOpenGLContext::EndFrame()
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}

	const char* ImGuiOpenGLContext::GetClipboardText(void* user_data)
	{
		return glfwGetClipboardString(static_cast<GLFWwindow*>(user_data));
	}

	void ImGuiOpenGLContext::SetClipboardText(void* user_data, const char* text)
	{
		glfwSetClipboardString(static_cast<GLFWwindow*>(user_data), text);
	}

	void ImGuiOpenGLContext::KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mode)
	{
		ImGuiIO& io = ImGui::GetIO();

		if (action == GLFW_PRESS) {
			io.KeysDown[key] = true;
		}
		if (action == GLFW_RELEASE) {
			io.KeysDown[key] = false;
		}

		io.KeyCtrl = io.KeysDown[GLFW_KEY_LEFT_CONTROL] || io.KeysDown[GLFW_KEY_RIGHT_CONTROL];
		io.KeyShift = io.KeysDown[GLFW_KEY_LEFT_SHIFT] || io.KeysDown[GLFW_KEY_RIGHT_SHIFT];
		io.KeyAlt = io.KeysDown[GLFW_KEY_LEFT_ALT] || io.KeysDown[GLFW_KEY_RIGHT_ALT];
		io.KeySuper = io.KeysDown[GLFW_KEY_LEFT_SUPER] || io.KeysDown[GLFW_KEY_RIGHT_SUPER];
	}

	void ImGuiOpenGLContext::CharCallback(GLFWwindow* window, unsigned int c)
	{
		ImGuiIO& io = ImGui::GetIO();
		if (c > 0 && c < 0x10000) {
			io.AddInputCharacter((unsigned short)c);
		}
	}

	void ImGuiOpenGLContext::MouseButtonCallback(GLFWwindow* window, int button, int action, int modify)
	{
		if (action == GLFW_PRESS && button >= 0 && button < 3) {
			imguiOpenGLContext->mousePressed[button] = true;
		}
	}

	void ImGuiOpenGLContext::ScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
	{
		imguiOpenGLContext->fMouseWheel += (float)yoffset;
	}
}