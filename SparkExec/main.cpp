#include "JsonSerializer.h"
#include "Spark.h"
#include "Structs.h"
#include "Logging.h"

int main()
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
        spark::Spark::setup(variables);
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
