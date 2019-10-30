#include "pch.h"

#include "Component.h"
#include <rttr/registration>
#include "Camera.h"

TEST(ReflectionTest, ComponentPropertiesValid) {
	rttr::type type{ rttr::type::get<spark::Component>() };
	ASSERT_EQ(3, type.get_properties().size());
	ASSERT_STREQ("Component", type.get_name().cbegin());
	rttr::property activeProp {type.get_property("active")}, nameProp{ type.get_property("name") },
		gameObjectProp{ type.get_property("gameObject") };
	ASSERT_TRUE(activeProp.is_valid());
	ASSERT_TRUE(nameProp.is_valid());
	ASSERT_TRUE(gameObjectProp.is_valid());
	ASSERT_STREQ("bool", activeProp.get_type().get_name().cbegin());
}

TEST(ReflectionTest, CameraInheritsComponent) {
	rttr::type cameraType{ rttr::type::get<spark::Camera>() }, componentType{ rttr::type::get<spark::Component>() };
	ASSERT_EQ(1, cameraType.get_base_classes().size());
	ASSERT_STREQ("Component", cameraType.get_base_classes().begin()->get_name().cbegin());
	ASSERT_FALSE(componentType.get_derived_classes().empty());
}
TEST(ReflectionTest, CameraPropertiesValidAndAccessible) {
	spark::Camera camera{};
	rttr::type type{ camera.get_type() };
	ASSERT_EQ(14 + rttr::type::get<spark::Component>().get_properties().size(), type.get_properties().size());
	ASSERT_TRUE(type.get_property("cameraTarget").is_valid());
	ASSERT_TRUE(type.get_property("Position").is_valid());
	ASSERT_TRUE(type.get_property("Front").is_valid());
	ASSERT_TRUE(type.get_property("Up").is_valid());
	ASSERT_TRUE(type.get_property("Right").is_valid());
	ASSERT_TRUE(type.get_property("Yaw").is_valid());
	ASSERT_TRUE(type.get_property("Pitch").is_valid());
	ASSERT_TRUE(type.get_property("MovementSpeed").is_valid());
	ASSERT_TRUE(type.get_property("MouseSensitivity").is_valid());
	ASSERT_TRUE(type.get_property("Zoom").is_valid());
	ASSERT_TRUE(type.get_property("cameraMode").is_valid());
	ASSERT_TRUE(type.get_property("fov").is_valid());
	ASSERT_TRUE(type.get_property("zNear").is_valid());
	ASSERT_TRUE(type.get_property("zFar").is_valid());
	ASSERT_EQ(camera.getCameraTarget(), type.get_property_value("cameraTarget", camera).get_value<glm::vec3>());
	ASSERT_EQ(camera.getPosition(), type.get_property_value("Position", camera).get_value<glm::vec3>());
	ASSERT_EQ(camera.getFront(), type.get_property_value("Front", camera).get_value<glm::vec3>());
	ASSERT_EQ(camera.getUp(), type.get_property_value("Up", camera).get_value<glm::vec3>());
	ASSERT_EQ(camera.getRight(), type.get_property_value("Right", camera).get_value<glm::vec3>());
	ASSERT_EQ(camera.getYaw(), type.get_property_value("Yaw", camera).get_value<float>());
	ASSERT_EQ(camera.getPitch(), type.get_property_value("Pitch", camera).get_value<float>());
	ASSERT_EQ(camera.getMovementSpeed(), type.get_property_value("MovementSpeed", camera).get_value<float>());
	ASSERT_EQ(camera.getMouseSensitivity(), type.get_property_value("MouseSensitivity", camera).get_value<float>());
	ASSERT_EQ(camera.getZoom(), type.get_property_value("Zoom", camera).get_value<float>());
	ASSERT_EQ(camera.getCameraMode(), type.get_property_value("cameraMode", camera).get_value<spark::CameraMode>());
	ASSERT_EQ(camera.getFov(), type.get_property_value("fov", camera).get_value<float>());
	ASSERT_EQ(camera.getNearPlane(), type.get_property_value("zNear", camera).get_value<float>());
	ASSERT_EQ(camera.getFarPlane(), type.get_property_value("zFar", camera).get_value<float>());
}