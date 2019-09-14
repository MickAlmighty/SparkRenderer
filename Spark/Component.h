#pragma once
#include <iostream>
#include <GameObject.h>

class GameObject;
class Component
{
	std::weak_ptr<GameObject> gameObject;
public:
	std::string name = "Component";
	Component() = default;
	Component(std::string& componentName);
	std::shared_ptr<GameObject> getGameObject() const { return gameObject.lock(); }
	virtual ~Component() = default;
	virtual void update() = 0;
	virtual void fixedUpdate() = 0;
	virtual void setGameObject(std::shared_ptr<GameObject>& game_object) { gameObject = game_object; };
};

