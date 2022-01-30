#include "pch.h"

#include "OpenGLContext.hpp"
#include "utils/GlHandle.hpp"

class TextureHandleTest : public ::testing::Test
{
    spark::OpenGLContext oglContext{1280, 720, true, true};
};

TEST_F(TextureHandleTest, testTextureHandleCreation)
{
    GLuint handle;
    glCreateTextures(GL_TEXTURE_2D, 1, &handle);

    const spark::utils::TextureHandle textureHandle(handle);

    ASSERT_EQ(textureHandle.get(), handle);
}

TEST_F(TextureHandleTest, testCopyingTextureHandleIncreasesItsUseCount)
{
    GLuint handle;
    glCreateTextures(GL_TEXTURE_2D, 1, &handle);

    const spark::utils::TextureHandle textureHandle(handle);
    ASSERT_EQ(textureHandle.get(), handle);
    ASSERT_EQ(textureHandle.use_count(), 1);
    const auto textureHandle2 = textureHandle;
    ASSERT_EQ(textureHandle2.use_count(), 2);
}

TEST_F(TextureHandleTest, testDefaultConstruction)
{
    const spark::utils::TextureHandle textureHandle{};
    ASSERT_EQ(textureHandle.get(), 0);
    ASSERT_EQ(textureHandle.use_count(), 0);
}