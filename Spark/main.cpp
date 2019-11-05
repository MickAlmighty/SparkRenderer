#include <iostream>

#include "JsonSerializer.h"
#include "Spark.h"
#include "Structs.h"
#include "Logging.h"

int main() {
    std::shared_ptr<spark::InitializationVariables> variables = spark::JsonSerializer::getInstance()->load<spark::InitializationVariables>("settings.json");
    try {
        spark::Spark::setup(variables);
        spark::Spark::run();
        spark::Spark::clean();
    } catch (std::exception& e) {
        SPARK_ERROR("{}", e.what());
        return 1;
    }
    getchar();
    return 0;
}
