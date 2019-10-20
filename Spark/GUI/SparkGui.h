#ifndef SPARK_GUI_H
#define SPARK_GUI_H

#include "ActorAI.h"
#include "Component.h"
#include "DirectionalLight.h"
#include "GameObject.h"
#include "Mesh.h"
#include "MeshPlane.h"
#include "ModelMesh.h"
#include "TerrainGenerator.h"

namespace spark {

class SparkGui
{
public:
	static std::shared_ptr<Component> addComponent();
	static std::pair<std::string, std::vector<Mesh>> getMeshes();
	static Texture getTexture();
	static std::shared_ptr<Shader> getShader();

	template <typename T>
	static std::shared_ptr<T> getObject(const std::string&& variableName, std::shared_ptr<T> object)
	{
		if (object == nullptr)
		{
			std::string objectName = variableName + ": " + "nullptr";
			ImGui::Text(objectName.c_str());
		}

		if (object != nullptr)
		{
			if (Component* c = dynamic_cast<Component*>(object.get()); c != nullptr)
			{
				std::string objectName = variableName + ": " + c->name + " (" + typeid(c).name() + ")";
				ImGui::Text(objectName.c_str());
			}
			if (GameObject* g = dynamic_cast<GameObject*>(object.get()); g != nullptr)
			{
				std::string objectName = variableName + ": " + g->name + " (" + typeid(g).name() + ")";
				ImGui::Text(objectName.c_str());
			}
		}
		
		if (ImGui::BeginDragDropTarget())
		{
			if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("OBJECT_DRAG_AND_DROP"))
			{
				IM_ASSERT(payload->DataSize == sizeof(std::shared_ptr<GameObject>));
				std::shared_ptr<GameObject> gameObject = *static_cast<const std::shared_ptr<GameObject>*>(payload->Data);
				if(dynamic_cast<T*>(gameObject.get()) != nullptr)
				{
					return std::dynamic_pointer_cast<T>(gameObject);
				}

				return gameObject->getComponent<T>(); // return nullptr if there is not any type T component
			}
			ImGui::EndDragDropTarget();
		}
		return object;
	}
};

const static std::map<std::string, std::function<std::shared_ptr<Component>()>> componentCreation{
	{"ModelMesh", [] { return std::make_shared<ModelMesh>(); }},
	{"MeshPlane", [] { return std::make_shared<MeshPlane>(); }},
	{"TerrainGenerator", [] { return std::make_shared<TerrainGenerator>(); }},
	{"ActorAI", [] { return std::make_shared<ActorAI>(); }},
	{"DirectionalLight", [] {return std::make_shared<DirectionalLight>(); }}
};

}
#endif