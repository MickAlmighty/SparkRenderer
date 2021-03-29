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

TEST_F(BufferTest, testProperBindingAsignment) {
    SSBO ssbo1{};
    SSBO ssbo2{};
    SSBO ssbo3{};
    SSBO ssbo4{};

    std::set<std::uint32_t> bindings{ 0, 1, 2, 3 };
    ASSERT_EQ(bindings, SSBO::bindings);

    ssbo2.~SSBO();
    ssbo3.~SSBO();

    bindings = { 0, 3 };
    ASSERT_EQ(bindings, SSBO::bindings);

    SSBO ssbo5{};
    bindings = { 0, 1, 3 };
    ASSERT_EQ(bindings, SSBO::bindings);

    SSBO ssbo6;
    bindings = { 0, 1, 2, 3 };
    ASSERT_EQ(bindings, SSBO::bindings);

    SSBO ssbo7{};
    bindings = { 0, 1, 2, 3, 4 };
    ASSERT_EQ(bindings, SSBO::bindings);
}