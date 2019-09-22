#ifndef SPARK_GUI_H
#define SPARK_GUI_H
#include <Component.h>
#include "Mesh.h"
#include <functional>
#include "ModelMesh.h"
#include "MeshPlane.h"
#include "TerrainGenerator.h"
#include "ActorAI.h"

class SparkGui
{
public:
	static std::shared_ptr<Component> addComponent();
	static std::pair<std::string, std::vector<Mesh>> getMeshes();
	static Texture getTexture();
};

const static std::map<std::string, std::function<std::shared_ptr<Component>()>> componentCreation {
	{"ModelMesh", [] { return std::make_shared<ModelMesh>(); }},
	{"MeshPlane", [] { return std::make_shared<MeshPlane>(); }},
	{"TerrainGenerator", [] { return std::make_shared<TerrainGenerator>(); }},
	{"ActorAI", [] { return std::make_shared<ActorAI>(); }}
};

#endif