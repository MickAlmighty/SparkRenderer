#include "pch.h"

#include <filesystem>

#include "JsonSerializer.h"
#include "ResourceFactory.h"
#include "Spark.h"

TEST(ResourceFactoryTest, ResourceCreation)
{
    spark::InitializationVariables variables;
    try
    {
        variables = spark::JsonSerializer::getInstance()->load<spark::InitializationVariables>("settings.json");
    }
    catch (std::exception&)
    {
        variables.width = 1280;
        variables.height = 720;
        variables.pathToResources = R"(..\..\..\res)";
        variables.pathToModels = R"(..\..\..\res\models)";
        spark::JsonSerializer::getInstance()->save(variables, "settings.json");
    }

    using namespace spark::resourceManagement;
    std::vector<std::shared_ptr<Resource>> createdResources;

    const std::filesystem::path resPath(variables.pathToResources);
    for(const auto& path : std::filesystem::recursive_directory_iterator(resPath))
    {
        std::optional<std::shared_ptr<Resource>> optResource = ResourceFactory::createResource(path);
        if (optResource != std::nullopt && optResource.value() != nullptr)
        {
            createdResources.push_back(optResource.value());
        }
    }

    ASSERT_TRUE(!createdResources.empty());
}

TEST(ResourceFactoryTest, ResourceCreationAndLoading)
{
   /* spark::InitializationVariables variables;
    try
    {
        variables = spark::JsonSerializer::getInstance()->load<spark::InitializationVariables>("settings.json");
    }
    catch (std::exception&)
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
    catch (std::exception & e)
    {
        SPARK_ERROR("{}", e.what());
    }

    using namespace spark::resourceManagement;
    std::vector<std::shared_ptr<Resource>> createdResources;

    const std::filesystem::path resPath(R"(C:\Studia\Semestr6\SparkRenderer\res)");
    for (const auto path : std::filesystem::recursive_directory_iterator(resPath))
    {
        std::optional<std::shared_ptr<Resource>> optResource = ResourceFactory::createResource(path);
        if (optResource != std::nullopt && optResource.value() != nullptr)
        {
            createdResources.push_back(optResource.value());
        }
    }

    ASSERT_TRUE(!createdResources.empty());

    try
    {
        spark::Spark::clean();
    }
    catch (std::exception & e)
    {
        SPARK_ERROR("{}", e.what());
    }*/
}