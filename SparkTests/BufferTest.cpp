#include "pch.h"

#include "JsonSerializer.h"
#include "Spark.h"
#include "SparkConfig.hpp"

inline void initSparkAndOpenGL()
{
    spark::SparkConfig config;
    try
    {
        config = spark::JsonSerializer::getInstance()->load<spark::SparkConfig>("settings.json");
    }
    catch(std::exception&)
    {
        config.width = 1280;
        config.height = 720;
        config.pathToResources = R"(..\..\..\res)";
        config.pathToModels = R"(..\..\..\res\models)";
        spark::JsonSerializer::getInstance()->save(config, "settings.json");
    }

    try
    {
        spark::Spark::loadConfig(config);
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