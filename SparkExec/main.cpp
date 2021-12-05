#include "JsonSerializer.h"
#include "Spark.h"
#include "Logging.h"
#include "SparkConfig.hpp"

int main()
{
    spark::SparkConfig config{};
    try
    {
        config = spark::JsonSerializer().load<spark::SparkConfig>("settings.json");
    }
    catch(std::exception&)
    {
        spark::JsonSerializer().save(config, "settings.json");
    }

    try
    {
        spark::Spark::run(config);
    }
    catch(std::exception& e)
    {
        SPARK_ERROR("{}", e.what());
        return 1;
    }
    return 0;
}
