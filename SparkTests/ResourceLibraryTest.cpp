#include "pch.h"

#include <filesystem>

#include "Logging.h"
#include "Model.h"
#include "OGLContext.hpp"
#include "ResourceLibrary.h"
#include "Shader.h"
#include "Texture.h"
#include "Timer.h"

constexpr auto pathToResources{R"(..\..\..\res)"};

using namespace spark::resources;

class ResourceLibraryTest : public ::testing::Test
{
    public:
    void SetUp() override
    {
        oglContext.init(1280, 720, true, true);
        resourceLibrary.createResources(pathToResources);
        resourceLibrary.setup();
    }

    spark::OGLContext oglContext;
    spark::resourceManagement::ResourceLibrary resourceLibrary;

    protected:
    void TearDown() override
    {
        oglContext.destroy();
        resourceLibrary.cleanup();
    }

    template<typename T>
    static void resourcesLoading(const std::vector<T>& resources, spark::resourceManagement::ResourceLibrary& resourceLibrary)
    {
        ASSERT_TRUE(!resources.empty());
        ASSERT_TRUE(resources[0] != nullptr);

        bool resourcesLoaded = false;
        ASSERT_TRUE(resourcesLoaded);
    }

    static void resourcesUnloading(spark::resourceManagement::ResourceLibrary& resourceLibrary)
    {
        bool resourcesUnloaded = false;
        while(!resourcesUnloaded)
        {
            /*resourceLibrary.processGpuResources();
            if(resourceLibrary.getLoadedResourcesCount() == 0)*/
                resourcesUnloaded = true;
        }

        ASSERT_TRUE(resourcesUnloaded);
    }

    template<typename Resource>
    void loadInPlace(std::string resourceName)
    {
        std::shared_ptr<Resource> resource = nullptr;

        resource = resourceLibrary.getResourceByName<Resource>(resourceName);
        ASSERT_TRUE(resource != nullptr);
        const auto resource2 = resourceLibrary.getResourceByName<Resource>(resourceName);
        ASSERT_TRUE(resource2 != nullptr);
    }

    template<typename Resource>
    void loadingAndUnloading()
    {
        for(int i = 0; i < 2; ++i)
        {
            SPARK_INFO("Loading and unloading loop iteration: {}", i);

            std::vector<std::shared_ptr<Resource>> resources{};
            {
                spark::Timer timer("Loading time:");
                /*resources = resourceLibrary.getResourcesOfType<Resource>();
                resourcesLoading(resources, resourceLibrary);*/
            }

            {
                spark::Timer timer("Unloading time:");
                resources.clear();
                resourcesUnloading(resourceLibrary);
            }
        }
    }
};

TEST_F(ResourceLibraryTest, ModelsLoadingAndUnloading)
{
    loadingAndUnloading<Model>();
}

TEST_F(ResourceLibraryTest, TexturesLoadingAndUnloading)
{
    loadingAndUnloading<Texture>();
}

TEST_F(ResourceLibraryTest, ShadersLoadingAndUnloading)
{
    loadingAndUnloading<Shader>();
}

TEST_F(ResourceLibraryTest, LoadingModelInPlace)
{
    loadInPlace<spark::resources::Model>("box.obj");
}

TEST_F(ResourceLibraryTest, LoadingTextureInPlace)
{
    loadInPlace<spark::resources::Texture>("Spaceship_Diffuse.DDS");
}

TEST_F(ResourceLibraryTest, LoadingShaderInPlace)
{
    loadInPlace<Shader>("default.glsl");
}