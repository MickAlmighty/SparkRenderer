#include <EngineSystems/SparkRenderer.h>
#include <EngineSystems/ResourceManager.h>
#include <exception>
#include <iostream>
#include <ResourceLoader.h>
#include <HID.h>
#include <Spark.h>

GLFWwindow* SparkRenderer::window = nullptr;
std::map<ShaderType, std::list<std::function<void(std::shared_ptr<Shader>&)>>> SparkRenderer::renderQueue;

void APIENTRY glDebugOutput(GLenum source,
	GLenum type,
	GLuint id,
	GLenum severity,
	GLsizei length,
	const GLchar *message,
	const void *userParam)
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

SparkRenderer* SparkRenderer::getInstance()
{
	static SparkRenderer* spark_renderer = nullptr;
	if(spark_renderer == nullptr)
	{
		spark_renderer = new SparkRenderer();
	}
	return  spark_renderer;
}

void SparkRenderer::setup()
{
	initMembers();
}

void SparkRenderer::initOpenGL()
{
	if (!glfwInit())
	{
		throw std::exception("glfw init failed");
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	
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

	glfwSetErrorCallback(error_callback);
	glfwSetKeyCallback(window, HID::key_callback);
	glfwSetCursorPosCallback(window, HID::cursor_position_callback);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

#ifdef DEBUG
	glEnable(GL_DEBUG_OUTPUT);
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
	glDebugMessageCallback(glDebugOutput, nullptr);
#endif

	unsigned char pixels[16 * 16 * 4];
	memset(pixels, 0xff, sizeof(pixels));
	GLFWimage image;
	image.width = 16;
	image.height = 16;
	image.pixels = pixels;
	GLFWcursor* cursor = glfwCreateCursor(&image, 0, 0);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	ImGui::StyleColorsLight();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	const char* glsl_version = "#version 450";
	ImGui_ImplOpenGL3_Init(glsl_version);

	glfwSetCursor(window, cursor);

	glfwSwapInterval(0);
}

void SparkRenderer::createTexture(GLuint& texture, GLuint width, GLuint height, GLenum internalFormat, GLenum format,
	GLenum pixelFormat, GLenum textureWrapping, GLenum textureSampling)
{
	glCreateTextures(GL_TEXTURE_2D, 1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);

	glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, pixelFormat, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, textureSampling);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, textureSampling);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, textureWrapping);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, textureWrapping);
}

void SparkRenderer::initMembers()
{
	screenQuad.setup();
	mainShader = ResourceManager::getInstance()->getShader(DEFAULT_SHADER);
	screenShader = ResourceManager::getInstance()->getShader(SCREEN_SHADER);
	postprocessingShader = ResourceManager::getInstance()->getShader(POSTPROCESSING_SHADER);

	glCreateFramebuffers(1, &mainFramebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, mainFramebuffer);
	createTexture(colorTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
	createTexture(positionTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);
	createTexture(normalsTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, positionTexture, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, colorTexture, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, normalsTexture, 0);

	GLuint renderbuffer;
	glCreateRenderbuffers(1, &renderbuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, Spark::WIDTH, Spark::HEIGHT);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderbuffer);

	GLenum attachments[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
	glDrawBuffers(3, attachments);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		throw std::exception("Main framebuffer incomplete!");
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glCreateFramebuffers(1, &postprocessingFramebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, postprocessingFramebuffer);
	createTexture(postProcessingTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, postProcessingTexture, 0);
	GLenum attachments2[1] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, attachments2);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		throw std::exception("Postprocessing framebuffer incomplete!");
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SparkRenderer::renderPass()
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	auto camera = SceneManager::getInstance()->getCurrentScene()->getCamera();
	camera->ProcessKeyboard();
	camera->ProcessMouseMovement(HID::mouse.direction.x, -HID::mouse.direction.y);
	glBindFramebuffer(GL_FRAMEBUFFER, mainFramebuffer);
	glClearColor(0, 0, 0, 1);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	glm::mat4 view = camera->GetViewMatrix();
	glm::mat4 projection = glm::perspective(glm::radians(60.0f), (float)Spark::WIDTH / Spark::HEIGHT, 0.1f, 100.0f);


	std::shared_ptr<Shader> shader = mainShader.lock();
	shader->use();
	shader->setMat4("view", view);
	shader->setMat4("projection", projection);
	for(auto& function: renderQueue[DEFAULT_SHADER])
	{
		function(shader);
	}
	renderQueue[DEFAULT_SHADER].clear();

	postprocessingPass();
	renderToScreen();

	SceneManager::getInstance()->getCurrentScene()->drawGUI();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	glfwSwapBuffers(window);
}

void SparkRenderer::postprocessingPass()
{
	glBindFramebuffer(GL_FRAMEBUFFER, postprocessingFramebuffer);
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	postprocessingShader.lock()->use();
	postprocessingShader.lock()->setVec2("inversedScreenSize", { 1.0f / Spark::WIDTH, 1.0f / Spark::HEIGHT });

	glBindTextureUnit(0, colorTexture);

	screenQuad.draw();
}


void SparkRenderer::renderToScreen()
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	screenShader.lock()->use();

	glBindTextureUnit(0, postProcessingTexture);

	screenQuad.draw();
}

void SparkRenderer::cleanup()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwDestroyWindow(window);
	glfwTerminate();
}

bool SparkRenderer::isWindowOpened()
{
	return !glfwWindowShouldClose(window);
}

void SparkRenderer::error_callback(int error, const char* description)
{
	fprintf(stderr, "Error: %s/n", description);
}
