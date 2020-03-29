#include "pch.h"

#include <filesystem>
#include <fstream>

#include "Logging.h"
#include "ResourceIdentifier.h"

const auto currentPath = std::filesystem::current_path();


TEST(ResourceIdentifierTest, ResourceIdentifierConstruction)
{
    using namespace spark::resourceManagement;
    std::ofstream file{ currentPath / "tmp.txt" };
    file.close();
    try
    {
        const auto resourcePath = currentPath / "tmp.txt";
        const ResourceIdentifier identifier(resourcePath);

        ASSERT_TRUE(resourcePath == identifier.getFullPath());
        ASSERT_TRUE(currentPath == identifier.getDirectoryPath());
        ASSERT_TRUE("tmp.txt" == identifier.getResourceName());
        ASSERT_STREQ("tmp", identifier.getResourceName(false).string().c_str());
        ASSERT_TRUE(".txt" == identifier.getResourceExtension());
    }
    catch (std::exception& e)
    {
        SPARK_ERROR("ResourceIdentifier construction failed: ", e.what());
    }
}

TEST(ResourceIdentifierTest, ResourceIdentifierChanges)
{
    using namespace spark::resourceManagement;
    std::ofstream file{ currentPath / "tmp.txt" };
    file.close();
    try
    {
        ResourceIdentifier identifier(currentPath / "tmp.txt");
        ASSERT_TRUE(currentPath == identifier.getDirectoryPath());
        ASSERT_TRUE("tmp.txt" == identifier.getResourceName());
        
        if (identifier.changeResourceName("tmp2.txt"))
        {
            ASSERT_STREQ("tmp2.txt", identifier.getResourceName().string().c_str());
        }

        if (identifier.changeResourceName("tmp.json"))
        {
            ASSERT_STREQ("tmp.json", identifier.getResourceName().string().c_str());
        }

        const auto parentDir = currentPath.parent_path();
        ASSERT_TRUE(identifier.changeResourceDirectory(parentDir));

        const auto nonExistentDir = currentPath / "dir";
        ASSERT_FALSE(identifier.changeResourceDirectory(nonExistentDir));

        ASSERT_TRUE(identifier.changeResourceDirectory(currentPath));

        std::filesystem::remove(currentPath / "tmp.txt");
    }
    catch (std::exception & e)
    {
        SPARK_ERROR("ResourceIdentifier construction failed: ", e.what());
    }
}

TEST(ResourceIdentifierTest, ResourceIdentifierOperatorsOverload)
{
    using namespace spark::resourceManagement;
    std::ofstream file{ currentPath / "a.txt" };
    std::ofstream file2{ currentPath / "b.txt" };
    file.close();
    file2.close();
    try
    {
        ResourceIdentifier identifier(currentPath / "a.txt");
        ResourceIdentifier identifier2(currentPath / "b.txt");

        ASSERT_TRUE(identifier < identifier2);
        ASSERT_FALSE(identifier == identifier2);

        std::filesystem::remove(identifier.getFullPath());
        std::filesystem::remove(identifier2.getFullPath());
    }
    catch (std::exception & e)
    {
        SPARK_ERROR("ResourceIdentifier construction failed: ", e.what());
    }

    
}