#include "pch.h"

#include <filesystem>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "MockSpark.hpp"
#include "OpenGLContext.hpp"
#include "ResourceFactory.h"
#include "Logging.h"

using ::testing::ReturnRef;

namespace spark::resourceManagement
{
class ResourceFactoryTest : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        SparkInstanceInjector::injectInstance(&sparkMock);
        EXPECT_CALL(sparkMock, getResourceLibrary()).WillRepeatedly(ReturnRef(resourceLibrary));
    }

    private:
    ResourceLibrary resourceLibrary = ResourceLibrary(utility::findFileOrDirectory("sparkData"));
    MockSpark sparkMock;
    spark::OpenGLContext oglContext{1280, 720, true, true};
};

TEST_F(ResourceFactoryTest, AllCreatedResourcesAreValid)
{
    std::vector<std::shared_ptr<Resource>> createdResources;

    SPARK_INFO(std::filesystem::current_path().string());

    const std::filesystem::path resPath = utility::findFileOrDirectory("sparkData");
    for(const auto& pathMaybeInvalid : std::filesystem::recursive_directory_iterator(resPath))
    {
        if(!pathMaybeInvalid.is_regular_file())
            continue;

        const auto& validFilePath = pathMaybeInvalid.path();
        if(const bool fileWithValidExtension = ResourceFactory::isExtensionSupported(validFilePath); fileWithValidExtension)
        {
            const auto sceneExts = ResourceFactory::supportedSceneExtensions();
            if(const auto isScene = std::find(sceneExts.cbegin(), sceneExts.cend(), validFilePath.extension().string()); isScene != sceneExts.cend())
            {
                continue;
            }

            const std::shared_ptr<Resource> resource = ResourceFactory::loadResource(resPath, validFilePath.lexically_relative(resPath));
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
    const std::shared_ptr<Resource> resource =
        ResourceFactory::loadResource("", "tmp123.obj");
    ASSERT_TRUE(resource == nullptr);
}

TEST_F(ResourceFactoryTest, CreatingResourceFromFileWithUnsuportedExtensionReturnsNullptr)
{
    const std::shared_ptr<Resource> resource =
        ResourceFactory::loadResource("", "tmp123.zip");
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
        ResourceFactory::supportedShaderExtensions().size() + ResourceFactory::supportedSceneExtensions().size() +
        ResourceFactory::supportedAnimationExtensions().size();
    ASSERT_EQ(numberOfAllSupportedExtensions, summedNumberOfSupportedExtensions);
}

}  // namespace spark::resourceManagement