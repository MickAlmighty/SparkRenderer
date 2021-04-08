#include "pch.h"

#include <filesystem>

#include "OGLContext.hpp"
#include "ResourceFactory.h"

class ResourceFactoryTest : public ::testing::Test
{
    void SetUp() override
    {
        oglContext.init(1280, 720, true, true);
    }

    spark::OGLContext oglContext;

    protected:
    void TearDown() override
    {
        oglContext.destroy();
    }
};
TEST_F(ResourceFactoryTest, AllCreatedResourcesAreValid)
{
    using namespace spark::resourceManagement;
    std::vector<std::shared_ptr<Resource>> createdResources;

    const std::filesystem::path resPath(R"(..\..\..\res)");
    for(const auto& path : std::filesystem::recursive_directory_iterator(resPath))
    {
        const std::shared_ptr<Resource> resource = ResourceFactory::createResource(path);
        if(resource != nullptr)
        {
            createdResources.push_back(resource);
        }
    }

    ASSERT_TRUE(!createdResources.empty());

    const auto allValid = std::none_of(createdResources.begin(), createdResources.end(), [](const auto& p) { return p == nullptr; });
    ASSERT_TRUE(allValid);
}

TEST_F(ResourceFactoryTest, CreatingResourceFromNonexistentFileReturnsNullptr)
{
    using namespace spark::resourceManagement;
    const std::shared_ptr<Resource> resource = ResourceFactory::createResource("tmp123.obj");
    ASSERT_TRUE(resource == nullptr);
}

TEST_F(ResourceFactoryTest, CreatingResourceFromFileWithUnsuportedExtensionReturnsNullptr)
{
    using namespace spark::resourceManagement;

    const std::shared_ptr<Resource> resource = ResourceFactory::createResource("tmp123.zip");
    ASSERT_TRUE(resource == nullptr);
}

TEST_F(ResourceFactoryTest, ValidFilenameHasSupportedExtension)
{
    ASSERT_TRUE(spark::resourceManagement::ResourceFactory::isExtensionSupported("tmp.obj"));
}

TEST_F(ResourceFactoryTest, InvalidFilenameHasUnsupportedExtension)
{
    ASSERT_FALSE(spark::resourceManagement::ResourceFactory::isExtensionSupported("tmp.gif"));
}

TEST_F(ResourceFactoryTest, InvalidFilenameWithoutExtensionReturnsFalse)
{
    ASSERT_FALSE(spark::resourceManagement::ResourceFactory::isExtensionSupported("tmp"));
}