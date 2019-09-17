#include <GUI/ImGui/imgui.h>
#include <Spark.h>
#include <Component.h>
#include "Mesh.h"

class SparkGui
{
public:
	static void drawMainMenuGui();
	static void drawSparkSettings(bool *p_open);
	static std::shared_ptr<Component> addComponent();
	static std::vector<Mesh> getMeshes();
};


const static std::vector<std::string> componentTypes{
	"ModelMesh",
	"Mockup"
};
