#include <GUI/ImGui/imgui.h>
#include <Spark.h>
#include <Component.h>
#include "Mesh.h"
#include "ModelMesh.h"

class SparkGui
{
public:
	static std::shared_ptr<Component> addComponent();
	static std::vector<Mesh> getMeshes();
};

const static std::map<std::string, std::function<std::shared_ptr<Component>()>> componentCreation {
	{"ModelMesh", []{ return std::make_shared<ModelMesh>(); }}
};