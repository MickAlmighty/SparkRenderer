#include "pch.h"

#include <filesystem>

#include "OGLContext.hpp"
#include "ResourceFactory.h"

namespace spark::resourceManagement
{
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
    std::vector<std::shared_ptr<Resource>> createdResources;

    const std::filesystem::path resPath(R"(..\..\..\res)");
    for(const auto& pathMaybeInvalid : std::filesystem::recursive_directory_iterator(resPath))
    {
        if (!pathMaybeInvalid.is_regular_file())
            continue;

        const bool fileWithValidExtension = ResourceFactory::isExtensionSupported(pathMaybeInvalid.path());
        if(fileWithValidExtension)
        {
            const std::shared_ptr<Resource> resource = ResourceFactory::createResource(pathMaybeInvalid);
            if(resource != nullptr)
            {
                createdResources.push_back(resource);
            }
            else
            {
                ASSERT_TRUE(false);
            }
        }
    }

    ASSERT_TRUE(!createdResources.empty());

    const auto allValid = std::none_of(createdResources.begin(), createdResources.end(), [](const auto& p) { return p == nullptr; });
    ASSERT_TRUE(allValid);
}

TEST_F(ResourceFactoryTest, CreatingResourceFromNonexistentFileReturnsNullptr)
{
    const std::shared_ptr<Resource> resource = ResourceFactory::createResource("tmp123.obj");
    ASSERT_TRUE(resource == nullptr);
}

TEST_F(ResourceFactoryTest, CreatingResourceFromFileWithUnsuportedExtensionReturnsNullptr)
{
    const std::shared_ptr<Resource> resource = ResourceFactory::createResource("tmp123.zip");
    ASSERT_TRUE(resource == nullptr);
}

TEST_F(ResourceFactoryTest, ValidFilenameHasSupportedExtension)
{
    ASSERT_TRUE(ResourceFactory::isExtensionSupported("tmp.obj"));
}

TEST_F(ResourceFactoryTest, InvalidFilenameHasUnsupportedExtension)
{
    ASSERT_FALSE(ResourceFactory::isExtensionSupported("tmp.gif"));
}

TEST_F(ResourceFactoryTest, InvalidFilenameWithoutExtensionReturnsFalse)
{
    ASSERT_FALSE(ResourceFactory::isExtensionSupported("tmp"));
}

TEST_F(ResourceFactoryTest, NumberOfSupportedExtensionsForDistinctResourcesAreSummingToNumberOfAllSupportedExtensions)
{
    const auto numberOfAllSupportedExtensions = ResourceFactory::supportedExtensions().size();
    const auto summedNumberOfSupportedExtensions =
        ResourceFactory::supportedModelExtensions().size() + ResourceFactory::supportedTextureExtensions().size() +
        ResourceFactory::supportedShaderExtensions().size() + ResourceFactory::supportedSceneExtensions().size();
    ASSERT_EQ(numberOfAllSupportedExtensions, summedNumberOfSupportedExtensions);
}

}  // namespace spark::resourceManagement