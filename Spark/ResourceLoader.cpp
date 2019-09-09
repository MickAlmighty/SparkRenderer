#include "ResourceLoader.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "Structs.h"

ResourceLoader::ResourceLoader()
{
}


ResourceLoader::~ResourceLoader()
{
}

std::unique_ptr<Mesh> ResourceLoader::loadMesh(aiMesh* assimpMesh)
{
	std::vector<Vertex> vertices(assimpMesh->mNumVertices);
	
	for(int i = 0; i < assimpMesh->mNumVertices; i++)
	{
		Vertex v{};

		v.pos.x = assimpMesh->mVertices[i].x;
		v.pos.y = assimpMesh->mVertices[i].y;
		v.pos.z = assimpMesh->mVertices[i].z;
		
		if(assimpMesh->HasNormals())
		{
			v.normal.x = assimpMesh->mNormals[i].x;
			v.normal.y = assimpMesh->mNormals[i].y;
			v.normal.z = assimpMesh->mNormals[i].z;
		}
		
		if(assimpMesh->HasTextureCoords(0))
		{
			v.texCoords.x = assimpMesh->mTextureCoords[0][i].x;
			v.texCoords.y = assimpMesh->mTextureCoords[0][i].y;
		}
		else
			v.texCoords = glm::vec2(0.0f, 0.0f);

		vertices[i] = v;
	}

	std::vector<unsigned int> indices;
	for (unsigned int i = 0; i < assimpMesh->mNumFaces; i++)
	{
		aiFace face = assimpMesh->mFaces[i];
		for (unsigned int j = 0; j < face.mNumIndices; j++)
			indices.push_back(face.mIndices[j]);
	}

	return std::make_unique<Mesh>(vertices, indices);
}

Model* ResourceLoader::loadModel(std::string&& path)
{
	Assimp::Importer importer;
	const aiScene *scene = importer.ReadFile(path, aiProcessPreset_TargetRealtime_Fast | aiProcess_FlipUVs);

	if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
	{
		throw std::exception(importer.GetErrorString());
	}

	std::vector<std::unique_ptr<Mesh>> meshes;

	for(int i = 0; i < scene->mNumMeshes; i++)
	{
		meshes.push_back(loadMesh(scene->mMeshes[i]));
	}

	return new Model(meshes);
}
