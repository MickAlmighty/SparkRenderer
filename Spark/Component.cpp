#include "Component.h"

#include <iostream>

namespace spark
{
	Component::Component(std::string& componentName)
	{
		name = componentName;
	}

	Component::~Component()
	{
#ifdef DEBUG
		std::cout << "Component: " + name << " destroyed!" << std::endl;
#endif
	}
}
