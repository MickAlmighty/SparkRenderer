#include "pch.h"

#include <filesystem>

#include "JsonSerializer.h"
#include "Model.h"
#include "ResourceLibrary.h"
#include "Shader.h"
#include "Spark.h"
#include "Texture.h"
#include "Timer.h"


inline void initSparkAndOpenGL()
{
	spark::InitializationVariables variables;
	try
	{
		variables = spark::JsonSerializer::getInstance()->load<spark::InitializationVariables>("settings.json");
	}
	catch (std::exception&)
	{
		variables.width = 1280;
		variables.height = 720;
		variables.pathToResources = R"(..\..\..\res)";
		variables.pathToModels = R"(..\..\..\res\models)";
		spark::JsonSerializer::getInstance()->save(variables, "settings.json");
	}

	try
	{
		spark::Spark::setInitVariables(variables);
		spark::Spark::initOpenGL();

	}
	catch (std::exception & e)
	{
		SPARK_ERROR("{}", e.what());
	}
}

inline void cleanupSpark()
{
	try
	{
		spark::Spark::clean();
	}
	catch (std::exception & e)
	{
		SPARK_ERROR("{}", e.what());
	}
}

template <typename T>
inline void resourcesLoading(const std::vector<T>& resources, spark::resourceManagement::ResourceLibrary& resourceLibrary)
{
	ASSERT_TRUE(!resources.empty());
	ASSERT_TRUE(resources[0] != nullptr);
	ASSERT_TRUE(resources[0]->isResourceReady() == false);

	bool resourcesLoaded = false;
	while (!resourcesLoaded)
	{
		resourceLibrary.processGpuResources();
		resourcesLoaded = std::all_of(resources.begin(), resources.end(), [](const std::shared_ptr<spark::resourceManagement::Resource>& resource)
			{
				return resource->isResourceReady();
			});
	}
	ASSERT_TRUE(resourcesLoaded);
}

inline void resourcesUnloading(spark::resourceManagement::ResourceLibrary& resourceLibrary)
{
	bool resourcesUnloaded = false;
	while (!resourcesUnloaded)
	{
		resourceLibrary.processGpuResources();
		if (resourceLibrary.getLoadedResourcesCount() == 0)
			resourcesUnloaded = true;
	}

	ASSERT_TRUE(resourcesUnloaded);
}

TEST(ResourceLibraryTest, ModelsLoadingAndUnloading)
{
	initSparkAndOpenGL();

	using namespace spark::resourceManagement;

	ResourceLibrary resourceLibrary = ResourceLibrary();
	resourceLibrary.createResources(spark::Spark::pathToResources);
	resourceLibrary.setup();

	for (int i = 0; i < 5; ++i)
	{
		SPARK_INFO("Loading and unloading loop iteration: {}", i);
        
		std::vector<std::shared_ptr<spark::resources::Model>> resources;
		{
			spark::Timer timer("Loading models time:");
			resources = resourceLibrary.getResourcesOfType<spark::resources::Model>();
			resourcesLoading(resources, resourceLibrary);
		}
		
		{
			spark::Timer timer("Unloading models time:");
			resources.clear();
			resourcesUnloading(resourceLibrary);
		}
	}

	resourceLibrary.cleanup();

	cleanupSpark();
}

TEST(ResourceLibraryTest, TexturesLoadingAndUnloading)
{
	initSparkAndOpenGL();

	using namespace spark::resourceManagement;

	ResourceLibrary resourceLibrary = ResourceLibrary();
	resourceLibrary.createResources(spark::Spark::pathToResources);
	resourceLibrary.setup();

	for (int i = 0; i < 5; ++i)
	{
		SPARK_INFO("Loading and unloading loop iteration: {}", i);
		std::vector<std::shared_ptr<spark::resources::Texture>> resources;
		{
			spark::Timer timer("Loading textures time:");
			resources = resourceLibrary.getResourcesOfType<spark::resources::Texture>();
			resourcesLoading(resources, resourceLibrary);
		}

		{
			spark::Timer timer("Unloading textures time:");
			resources.clear();
			resourcesUnloading(resourceLibrary);
		}
	}
	
	resourceLibrary.cleanup();

	cleanupSpark();
}

TEST(ResourceLibraryTest, ShadersLoadingAndUnloading)
{
	initSparkAndOpenGL();

	using namespace spark::resourceManagement;

	ResourceLibrary resourceLibrary = ResourceLibrary();
	resourceLibrary.createResources(spark::Spark::pathToResources);
	resourceLibrary.setup();

	for (int i = 0; i < 5; ++i)
	{
		SPARK_INFO("Loading and unloading loop iteration: {}", i);

		std::vector<std::shared_ptr<spark::resources::Shader>> resources;
		{
			spark::Timer timer("Loading shaders time:");
			resources = resourceLibrary.getResourcesOfType<spark::resources::Shader>();
			resourcesLoading(resources, resourceLibrary);
		}

		{
			spark::Timer timer("Unloading shaders time:");
			resources.clear();
			resourcesUnloading(resourceLibrary);
		}
	}

	resourceLibrary.cleanup();

	cleanupSpark();
}

TEST(ResourceLibraryTest, LoadingModelInPlace)
{
	initSparkAndOpenGL();

	using namespace spark::resourceManagement;

	ResourceLibrary resourceLibrary = ResourceLibrary();
	resourceLibrary.createResources(spark::Spark::pathToResources);
	resourceLibrary.setup();

	std::shared_ptr<spark::resources::Model> model = nullptr;

	ASSERT_TRUE(model == nullptr);
	model = resourceLibrary.getResourceByNameWithOptLoad<spark::resources::Model>("box.obj");
	ASSERT_TRUE(model->isResourceReady());

	const auto model2 = resourceLibrary.getResourceByNameWithOptLoad<spark::resources::Model>("box.obj");
	ASSERT_TRUE(model2->isResourceReady());

	resourceLibrary.cleanup();

	cleanupSpark();
}

TEST(ResourceLibraryTest, LoadingTextureInPlace)
{
	initSparkAndOpenGL();

	using namespace spark::resourceManagement;

	ResourceLibrary resourceLibrary = ResourceLibrary();
	resourceLibrary.createResources(spark::Spark::pathToResources);
	resourceLibrary.setup();

	std::shared_ptr<spark::resources::Texture> texture = nullptr;

	ASSERT_TRUE(texture == nullptr);
	texture = resourceLibrary.getResourceByNameWithOptLoad<spark::resources::Texture>("Spaceship_Diffuse.DDS");
	ASSERT_TRUE(texture->isResourceReady());

	const auto texture2 = resourceLibrary.getResourceByNameWithOptLoad<spark::resources::Texture>("Spaceship_Diffuse.DDS");
	ASSERT_TRUE(texture2->isResourceReady());

	resourceLibrary.cleanup();

	cleanupSpark();
}

TEST(ResourceLibraryTest, LoadingShaderInPlace)
{
	initSparkAndOpenGL();

	using namespace spark::resourceManagement;

	ResourceLibrary resourceLibrary = ResourceLibrary();
	resourceLibrary.createResources(spark::Spark::pathToResources);
	resourceLibrary.setup();

	std::shared_ptr<spark::resources::Shader> shader = nullptr;

	ASSERT_TRUE(shader == nullptr);
	shader = resourceLibrary.getResourceByNameWithOptLoad<spark::resources::Shader>("default.glsl");
	ASSERT_TRUE(shader->isResourceReady());

    const auto shader2 = resourceLibrary.getResourceByNameWithOptLoad<spark::resources::Shader>("default.glsl");
	ASSERT_TRUE(shader2->isResourceReady());

	resourceLibrary.cleanup();

	cleanupSpark();
}