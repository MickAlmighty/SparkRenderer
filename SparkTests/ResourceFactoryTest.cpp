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

    const std::filesystem::path root = utility::findFileOrDirectory("sparkData");

    private:
    ResourceLibrary resourceLibrary = ResourceLibrary(root);
    MockSpark sparkMock;
    spark::OpenGLContext oglContext{1280, 720, true, true};
};

TEST_F(ResourceFactoryTest, CreatingResourceFromNonexistentFileReturnsNullptr)
{
    const std::shared_ptr<Resource> resource = ResourceFactory::loadResource("", "tmp123.obj");
    ASSERT_TRUE(resource == nullptr);
}

TEST_F(ResourceFactoryTest, CreatingResourceFromFileWithUnsuportedExtensionReturnsNullptr)
{
    const std::shared_ptr<Resource> resource = ResourceFactory::loadResource("", "tmp123.zip");
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