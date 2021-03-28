#include "pch.h"

#include <filesystem>

#include "JsonSerializer.h"
#include "ResourceFactory.h"
#include "Spark.h"
#include "SparkConfig.hpp"

TEST(ResourceFactoryTest, ResourceCreation)
{
    using namespace spark::resourceManagement;
    std::vector<std::shared_ptr<Resource>> createdResources;

    const std::filesystem::path resPath(R"(..\..\..\res)");
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