#pragma once
#include <iostream>

class Component
{
	
public:
	std::string name = "Component";

	Component() = default;
	Component(std::string& componentName);
	virtual ~Component() = default;
	virtual void update() = 0;
	virtual void fixedUpdate() = 0;
};

