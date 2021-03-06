#include "pch.h"

#include "Buffer.hpp"
#include "OGLContext.hpp"

class BufferTest : public ::testing::Test
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

TEST_F(BufferTest, testProperBindingAsignment)
{
    SSBO ssbo1{};

    {
        SSBO ssbo2{};
        SSBO ssbo3{};
        SSBO ssbo4{};
        ASSERT_EQ(ssbo1.binding, 0);
        ASSERT_EQ(ssbo2.binding, 1);
        ASSERT_EQ(ssbo3.binding, 2);
        ASSERT_EQ(ssbo4.binding, 3);
    }

    SSBO ssbo5{};
    ASSERT_EQ(ssbo5.binding, 1);
}