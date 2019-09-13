#pragma once
#include <json/value.h>
#include <filesystem>

class JsonSerializer
{
	JsonSerializer();
	~JsonSerializer();
public:
	static void writeToFile(std::filesystem::path&& filePath, Json::Value&& root);
	static Json::Value readFromFile(std::filesystem::path&& filePath);
};

