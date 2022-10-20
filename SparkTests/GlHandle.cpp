#include "pch.h"

#include "OpenGLContext.hpp"
#include "utils/GlHandle.hpp"

class TextureHandleTest : public ::testing::Test
{
    spark::OpenGLContext oglContext{1280, 720, true, true};
};

TEST_F(TextureHandleTest, TextureHandleCreation)
{
    GLuint handle;
    glCreateTextures(GL_TEXTURE_2D, 1, &handle);

    const spark::utils::TextureHandle textureHandle(handle);

    ASSERT_EQ(textureHandle.get(), handle);
}

TEST_F(TextureHandleTest, CopyingTextureHandleIncreasesItsUseCount)
{
    GLuint handle;
    glCreateTextures(GL_TEXTURE_2D, 1, &handle);

    const spark::utils::TextureHandle textureHandle(handle);
    ASSERT_EQ(textureHandle.get(), handle);
    ASSERT_EQ(textureHandle.use_count(), 1);
    const auto textureHandle2 = textureHandle;
    ASSERT_EQ(textureHandle2.use_count(), 2);
}

TEST_F(TextureHandleTest, DefaultConstruction)
{
    const spark::utils::TextureHandle textureHandle{};
    ASSERT_EQ(textureHandle.get(), 0);
    ASSERT_EQ(textureHandle.use_count(), 0);
}

TEST_F(TextureHandleTest, DefaultConstructionOfUniqueTextureHandle)
{
    const spark::utils::UniqueTextureHandle textureHandle{};
    ASSERT_EQ(textureHandle.get(), 0);
}

TEST_F(TextureHandleTest, ConstructionOfSharedGlHandleFromUniqueGlHandle)
{
    GLuint handle;
    glCreateTextures(GL_TEXTURE_2D, 1, &handle);

    const spark::utils::TextureHandle textureHandle{spark::utils::UniqueTextureHandle{handle}};
    ASSERT_EQ(textureHandle.get(), handle);
}

TEST_F(TextureHandleTest, ConstructionOfSharedGlHandleByAssignmentOfUniqueGlHandle)
{
    GLuint handle;
    glCreateTextures(GL_TEXTURE_2D, 1, &handle);

    const spark::utils::TextureHandle textureHandle = spark::utils::UniqueTextureHandle{handle};
    ASSERT_EQ(textureHandle.get(), handle);
}