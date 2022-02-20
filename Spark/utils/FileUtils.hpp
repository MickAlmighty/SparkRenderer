#pragma once
#include <filesystem>

namespace spark::utils
{
void write_text_to_file(const std::filesystem::path& path, const std::string& text);
std::string load_text_from_file(const std::filesystem::path& path);
}  // namespace spark::utils