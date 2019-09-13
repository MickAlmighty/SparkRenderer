#include "JsonSerializer.h"
#include <iostream>
#include <fstream>
#include <json/reader.h>
#include <json/writer.h>


JsonSerializer::JsonSerializer()
{
}


JsonSerializer::~JsonSerializer()
{
}

void JsonSerializer::writeToFile(std::filesystem::path&& filePath, Json::Value&& root)
{
	Json::StreamWriterBuilder builder;
	std::ofstream file(filePath);
	Json::StreamWriter* writer = builder.newStreamWriter();
	writer->write(root, &file);
}

Json::Value JsonSerializer::readFromFile(std::filesystem::path&& filePath)
{
	Json::Value root;
	std::ifstream file("settings.json", std::ios::in | std::ios::binary | std::ios::ate);
	if (file.is_open())
	{
		auto size = file.tellg();
		char* data = new char[size];
		file.seekg(0, std::ios::beg);
		file.read(data, size);
		
		file.close();

		Json::CharReaderBuilder builder;
		Json::CharReader* reader = builder.newCharReader();

		std::string errors;
		reader->parse(data, data + size, &root, &errors);
		delete[] data;
	}
	file.close();
	return root;
}
