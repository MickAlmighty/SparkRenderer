#include "pch.h"

#include <filesystem>
#include <fstream>

#include "Logging.h"
#include "ResourceIdentifier.h"

TEST(ResourceIdentifierTest, ResourceIdentifierConstruction)
{
    using namespace spark::resourceManagement;
    const auto currentPath = std::filesystem::current_path();
    const ResourceIdentifier identifier(currentPath, "tmp.txt");

    ASSERT_TRUE(currentPath / "tmp.txt" == identifier.getFullPath());
    ASSERT_TRUE("tmp.txt" == identifier.getRelativePath());
    ASSERT_TRUE("tmp.txt" == identifier.getResourceName());
    ASSERT_STREQ("tmp", identifier.getResourceName(false).string().c_str());
    ASSERT_TRUE(".txt" == identifier.getResourceExtension());
}

TEST(ResourceIdentifierTest, ResourceIdentifierOperatorsOverload)
{
    using namespace spark::resourceManagement;
    const auto currentPath = std::filesystem::current_path();
    const ResourceIdentifier identifier(currentPath, "a.txt");
    const ResourceIdentifier identifier2(currentPath, "b.txt");

    ASSERT_TRUE(identifier < identifier2);
    ASSERT_FALSE(identifier == identifier2);

    std::filesystem::remove(identifier.getFullPath());
    std::filesystem::remove(identifier2.getFullPath());
}