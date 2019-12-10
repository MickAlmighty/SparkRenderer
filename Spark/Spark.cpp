#include "Spark.h"

#include <iostream>

#include <GUI/ImGui/imgui.h>
#include <GUI/ImGui/imgui_impl_glfw.h>
#include <GUI/ImGui/imgui_impl_opengl3.h>

#include "Clock.h"
#include "CUDA/PathfindingManager.h"
#include "EngineSystems/SparkRenderer.h"
#include "EngineSystems/ResourceManager.h"
#include "EngineSystems/SceneManager.h"
#include "HID.h"
#include "JsonSerializer.h"
#include "Timer.h"

namespace spark {

	void Spark::setup(InitializationVariables& variables)
	{
		PROFILE_FUNCTION();
		WIDTH = variables.width;
		HEIGHT = variables.height;
		pathToModelMeshes = variables.pathToModels;
		pathToResources = variables.pathToResources;
		vsync = variables.vsync;

		initOpenGL();
		ResourceManager::getInstance()->loadResources();
		SceneManager::getInstance()->setup();
		PathFindingManager::getInstance()->loadMap(variables.mapPath);
		SparkRenderer::getInstance()->setup();
	}

	void APIENTRY glDebugOutput(GLenum source,
		GLenum type,
		GLuint id,
		GLenum severity,
		GLsizei length,
		const GLchar *message,
		const void *userParam);

	void Spark::initOpenGL()
	{
		if (!glfwInit())
		{
			throw std::exception("glfw init failed");
		}

		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		//glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		window = glfwCreateWindow(Spark::WIDTH, Spark::HEIGHT, "Spark", NULL, NULL);
		if (!window)
		{
			throw std::exception("Window creation failed");
		}

		glfwMakeContextCurrent(window);

		if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
		{
			throw std::exception("Failed to initialize OpenGL loader!");
		}

		glfwSetKeyCallback(window, HID::key_callback);
		glfwSetCursorPosCallback(window, HID::cursor_position_callback);
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	#ifdef DEBUG
		glEnable(GL_DEBUG_OUTPUT);
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
		glDebugMessageCallback(glDebugOutput, nullptr);
	#endif

		glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAnisotropicFiltering);
		glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void)io;
		ImGui::StyleColorsLight();

		ImGui_ImplGlfw_InitForOpenGL(window, true);
		const char* glsl_version = "#version 450";
		ImGui_ImplOpenGL3_Init(glsl_version);
		ImGui_ImplOpenGL3_NewFrame();

		if(vsync)
		{
			glfwSwapInterval(1);
		}
		else
		{
			glfwSwapInterval(0);
		}
	}

	void Spark::run()
	{
		while (!glfwWindowShouldClose(window) && runProgram)
		{
			Timer::capture = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) && glfwGetKey(window, GLFW_KEY_L);
			if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) && glfwGetKey(window, GLFW_KEY_H))
			{
				Spark::gui = false;
			}
			if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) && glfwGetKey(window, GLFW_KEY_G))
			{
				Spark::gui = true;
			}
			PROFILE_FUNCTION();
			Clock::tick();
			glfwPollEvents();
			SceneManager::getInstance()->update();
			PathFindingManager::getInstance()->findPaths();
			
			if (gui)
				sparkGui.drawGui();

			SparkRenderer::getInstance()->renderPass();
			HID::clearStates();
	#ifdef DEBUG
			//std::cout << Clock::getFPS() << std::endl;
	#endif
		}
	}

	void Spark::resizeWindow(GLuint width, GLuint height)
	{
		glfwSetWindowSize(window, width, height);
	}

	void Spark::clean()
	{
		SparkRenderer::getInstance()->cleanup();
		SceneManager::getInstance()->cleanup();
		ResourceManager::getInstance()->cleanup();
		destroyOpenGLContext();
	}

	void Spark::destroyOpenGLContext()
	{
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
		glfwDestroyWindow(window);
		glfwTerminate();
	}

	void APIENTRY glDebugOutput(GLenum source,
	GLenum type,
	GLuint id,
	GLenum severity,
	GLsizei length,
	const GLchar *message,
	const void *userParam)
	{
		// ignore non-significant error/warning codes
		//if (id == 0 || id == 131169 || id == 131185 || id == 131218 || id == 131204) return;
		if (id == 0 || id == 131185) return;

		std::cout << "---------------" << std::endl;
		std::cout << "Debug message id: (" << id << "): " << message << std::endl;

		switch (source)
		{
		case GL_DEBUG_SOURCE_API:             std::cout << "Source: API"; break;
		case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cout << "Source: Window System"; break;
		case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cout << "Source: Shader Compiler"; break;
		case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cout << "Source: Third Party"; break;
		case GL_DEBUG_SOURCE_APPLICATION:     std::cout << "Source: Application"; break;
		case GL_DEBUG_SOURCE_OTHER:           std::cout << "Source: Other"; break;
		} std::cout << std::endl;

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
		} std::cout << std::endl;

		switch (severity)
		{
		case GL_DEBUG_SEVERITY_HIGH:         std::cout << "Severity: high"; break;
		case GL_DEBUG_SEVERITY_MEDIUM:       std::cout << "Severity: medium"; break;
		case GL_DEBUG_SEVERITY_LOW:          std::cout << "Severity: low"; break;
		case GL_DEBUG_SEVERITY_NOTIFICATION: std::cout << "Severity: notification"; break;
		} std::cout << std::endl;
		std::cout << std::endl;
	}

}
