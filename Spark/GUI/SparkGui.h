#ifndef SPARK_GUI_H
#define SPARK_GUI_H
#include <Component.h>
#include "Mesh.h"
#include <functional>
#include "ModelMesh.h"

class SparkGui
{
public:
	static std::shared_ptr<Component> addComponent();
	static std::pair<std::string, std::vector<Mesh>> getMeshes();
};

const static std::map<std::string, std::function<std::shared_ptr<Component>()>> componentCreation {
	{"ModelMesh", []{ return std::make_shared<ModelMesh>(); }}
};

#endif