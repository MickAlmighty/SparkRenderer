#include "EngineSystems/ResourceManager.h"

#include "Mesh.h"
#include "ResourceLoader.h"
#include "Shader.h"
#include "Spark.h"
#include "Timer.h"
#include "Logging.h"

namespace spark {

	ResourceManager* ResourceManager::getInstance()
	{
		static auto resource_manager = new ResourceManager();
		return resource_manager;
	}

	void ResourceManager::addTexture(Texture tex)
	{
		const auto tex_it = std::find_if(std::begin(textures), std::end(textures), [&tex](const Texture& texture)
		{
			return texture.path == tex.path;
		});
		if (tex_it != std::end(textures))
		{
			textures.erase(tex_it);

		}
		textures.push_back(tex);
	}

	void ResourceManager::addCubemapTexturePath(const std::string& path)
	{
		cubemapTexturePaths.emplace_back(path);
	}

	Texture ResourceManager::findTexture(const std::string&& path) const
	{
		for (auto& tex_it : textures)
		{
			if (tex_it.path == path)
				return tex_it;
		}
        SPARK_ERROR("Texture from path '{}' was NOT found!", path);
		return {};
	}

	std::vector<Mesh> ResourceManager::findModelMeshes(const std::string& path) const
	{
		const auto& it = models.find(path);
		if (it != models.end())
		{
			return it->second;
		}
        SPARK_ERROR("Meshes for model from path '{}' were NOT found!", path);
		return {};
	}

    GLuint ResourceManager::getTextureId(const std::string& path) const {
        const auto& it
        {
            std::find_if(textures.begin(), textures.end(), [&](const Texture& tex) { return tex.getPath() == path; })
        };
        if(it != textures.end())
        {
            return it->getId();
        }
        SPARK_ERROR("ID for texture from path '{}' was NOT found!", path);
        return 0;
	}

std::vector<std::string> ResourceManager::getPathsToModels() const
	{
		std::vector<std::string> paths;
		for (auto& element : models)
		{
			paths.push_back(element.first);
		}
		return paths;
	}

	std::shared_ptr<Shader> ResourceManager::getShader(const ShaderType& type) const
	{
		const auto& it = shaders.find(type);
		if (it != shaders.end())
		{
			return it->second;
		}
        SPARK_ERROR("Cannot find shader! Probably shader was not loaded!");
		throw std::exception("Cannot find shader! Probably shader was not loaded!");
	}

	std::shared_ptr<Shader> ResourceManager::getShader(const std::string& name) const
	{
		const auto it = std::find_if(std::begin(shaders), std::end(shaders), [&name](const std::pair<ShaderType, std::shared_ptr<Shader>>& pair)
		{
			return pair.second->name == name;
		});
		if (it != std::end(shaders))
		{
			return it->second;
		}
        SPARK_ERROR("Shader with name '{}' was NOT found!", name);
		return nullptr;
	}

	std::vector<std::string> ResourceManager::getShaderNames() const
	{
		std::vector<std::string> shaderNames;
		shaderNames.reserve(shaders.size());
		for(const auto& [shaderEnum, shader_ptr] : shaders)
		{
			shaderNames.push_back(shader_ptr->name);
		}
		return shaderNames;
	}

	std::vector<Texture> ResourceManager::getTextures() const
	{
		return textures;
	}

	const std::vector<std::string>& ResourceManager::getCubemapTexturePaths() const
	{
		return cubemapTexturePaths;
	}

	void ResourceManager::drawGui()
	{
		
		static bool textureWindowOpened = false;
		if (ImGui::BeginMenu("ResourceManager"))
		{
			std::string menuName = "Resources";
			if (ImGui::BeginMenu(menuName.c_str()))
			{
				if(ImGui::MenuItem("Textures"))
				{
					textureWindowOpened = true;
				}
				ImGui::EndMenu();
			}
			ImGui::EndMenu();
		}

		if(textureWindowOpened)
		{
			if (ImGui::Begin("Textures", &textureWindowOpened, ImGuiWindowFlags_NoCollapse))
			{
				for (const auto& texture : textures)
				{
					ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(texture.ID)), ImVec2(200.0f, 200.0f), ImVec2(0, 1), ImVec2(1, 0));
					
					if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID))
					{
						ImGui::SetDragDropPayload("TEXTURE", &texture, sizeof(Texture));        // Set payload to carry the index of our item (could be anything)
						ImGui::Text("Getting texture %s", texture.path.c_str());
						ImGui::EndDragDropSource();
					}
					std::string s = std::to_string(texture.ID) + ": " + texture.path;
					ImGui::Text(s.c_str());
					ImGui::NewLine();
				}
				ImGui::End();
			}
		}
	}

	void ResourceManager::loadResources()
	{
		Timer timer("ResourceManager::loadResources");
		std::filesystem::path shaderDir = Spark::pathToResources;
		shaderDir.append("shaders\\");
		{
			Timer timer2("ResourceManager::loadResources -> shaders");
			shaders.emplace(ShaderType::DEFAULT_SHADER, std::make_shared<Shader>(shaderDir.string() + "default.glsl"));
			shaders.emplace(ShaderType::SCREEN_SHADER, std::make_shared<Shader>(shaderDir.string() + "screen.glsl"));
			shaders.emplace(ShaderType::POSTPROCESSING_SHADER, std::make_shared<Shader>(shaderDir.string() + "postprocessing.glsl"));
			shaders.emplace(ShaderType::LIGHT_SHADER, std::make_shared<Shader>(shaderDir.string() + "light.glsl"));
			shaders.emplace(ShaderType::MOTION_BLUR_SHADER, std::make_shared<Shader>(shaderDir.string() + "motionBlur.glsl"));
			shaders.emplace(ShaderType::EQUIRECTANGULAR_TO_CUBEMAP_SHADER, std::make_shared<Shader>(shaderDir.string() + "equirectangularToCubemap.glsl"));
			shaders.emplace(ShaderType::CUBEMAP_SHADER, std::make_shared<Shader>(shaderDir.string() + "cubemap.glsl"));
			shaders.emplace(ShaderType::IRRADIANCE_SHADER, std::make_shared<Shader>(shaderDir.string() + "irradiance.glsl"));
			shaders.emplace(ShaderType::PREFILTER_SHADER, std::make_shared<Shader>(shaderDir.string() + "prefilter.glsl"));
			shaders.emplace(ShaderType::BRDF_SHADER, std::make_shared<Shader>(shaderDir.string() + "brdf.glsl"));
		}
		
		{
			Timer timer3("ResourceManager::loadResources -> textures");
			textures = ResourceLoader::loadTextures(Spark::pathToResources);
		}

		{
			Timer timer3("ResourceManager::loadResources -> models");
			models = ResourceLoader::loadModels(Spark::pathToModelMeshes);
		}
	}

	void ResourceManager::cleanup()
	{
		for (auto& tex_it : textures)
		{
			glDeleteTextures(1, &tex_it.ID);
		}
		textures.clear();
		for (auto& model_it : models)
		{
			//std::vector<Mesh> meshes = model_it.second;
			for (Mesh& mesh : model_it.second)
			{
				mesh.cleanup();
			}
		}
		models.clear();
		shaders.clear();
	}

}
