#include "pch.h"

#include "JsonSerializer.h"
#include "Spark.h"

inline void initSparkAndOpenGL()
{
    spark::InitializationVariables variables;
    try
    {
        variables = spark::JsonSerializer::getInstance()->load<spark::InitializationVariables>("settings.json");
    }
    catch(std::exception&)
    {
        variables.width = 1280;
        variables.height = 720;
        variables.pathToResources = R"(..\..\..\res)";
        variables.pathToModels = R"(..\..\..\res\models)";
        spark::JsonSerializer::getInstance()->save(variables, "settings.json");
    }

    try
    {
        spark::Spark::setInitVariables(variables);
        spark::Spark::initOpenGL();
    }
    catch(std::exception& e)
    {
        SPARK_ERROR("{}", e.what());
    }
}

inline void cleanupSpark()
{
    try
    {
        spark::Spark::destroyOpenGLContext();
    }
    catch(std::exception& e)
    {
        SPARK_ERROR("{}", e.what());
    }
}

TEST(BufferBindingsTest, BufferTest)
{
    initSparkAndOpenGL();

    SSBO ssbo1{};
    SSBO ssbo2{};
    SSBO ssbo3{};
    SSBO ssbo4{};

    std::set<std::uint32_t> bindings{0, 1, 2, 3};
    ASSERT_EQ(bindings, SSBO::bindings);

    ssbo2.~SSBO();
    ssbo3.~SSBO();

    bindings = {0, 3};
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

    cleanupSpark();
}