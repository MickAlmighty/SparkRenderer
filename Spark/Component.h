#pragma once
#include <iostream>
#include "GameObject.h"

class GameObject;
class Component
{
	
public:
	std::string name = "Component";
	std::weak_ptr<GameObject> gameObject;
	Component() = default;
	Component(std::string& componentName);
	virtual ~Component() = default;
	virtual void update() = 0;
	virtual void fixedUpdate() = 0;
};

