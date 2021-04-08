#include "pch.h"

#include <filesystem>

#include "Logging.h"
#include "Model.h"
#include "OGLContext.hpp"
#include "ResourceLibrary.h"
#include "Shader.h"
#include "Texture.h"

constexpr auto pathToResources{R"(..\..\..\res)"};

using namespace spark::resources;

class ResourceLibraryTest : public ::testing::Test
{
    public:
    void SetUp() override
    {
        oglContext.init(1280, 720, true, true);
        resourceLibrary.setup(pathToResources);
    }

    spark::OGLContext oglContext;
    spark::resourceManagement::ResourceLibrary resourceLibrary;

    protected:
    void TearDown() override
    {
        oglContext.destroy();
        resourceLibrary.cleanup();
    }

    template<typename Resource>
    void loadInPlace(std::string resourceName)
    {
        std::shared_ptr<Resource> resource = resourceLibrary.getResourceByName<Resource>(resourceName);
        ASSERT_TRUE(resource != nullptr);
        const auto resource2 = resourceLibrary.getResourceByName<Resource>(resourceName);
        ASSERT_TRUE(resource2 != nullptr);
        ASSERT_TRUE(resource2.use_count() == 2);
    }
};

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