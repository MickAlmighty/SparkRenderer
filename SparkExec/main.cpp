#include "JsonSerializer.h"
#include "Spark.h"
#include "Logging.h"
#include "SparkConfig.hpp"

int main()
{
    spark::SparkConfig config{};
    try
    {
        config = spark::JsonSerializer::getInstance()->load<spark::SparkConfig>("settings.json");
    }
    catch(std::exception&)
    {
        spark::JsonSerializer::getInstance()->save(config, "settings.json");
    }

    try
    {
        spark::Spark::loadConfig(config);
        spark::Spark::setup();
        spark::Spark::run();
        spark::Spark::clean();
    }
    catch(std::exception& e)
    {
        SPARK_ERROR("{}", e.what());
        getchar();
        return 1;
    }
    return 0;
}
