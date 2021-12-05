#include "pch.h"

#include "Component.h"
#include "GameObject.h"

#include "glm/gtx/string_cast.hpp"

class TestComponent1 : public spark::Component
{
    public:
    TestComponent1() : spark::Component() {}
    void update() override {}
};

namespace spark
{
TEST(ComponentTest, settingGameObject)
{
    const auto component = std::make_shared<TestComponent1>();
    const auto gameObject = std::make_shared<GameObject>();

    ASSERT_EQ(component->getGameObject(), nullptr);
    component->setGameObject(gameObject);
    ASSERT_EQ(component->getGameObject(), gameObject);
}

TEST(ComponentTest, settingGameObjectToComponentWhichHasGameObjectAlreadyShouldRemoveComponentFromItsComponents)
{
    const auto component = std::make_shared<TestComponent1>();
    const auto gameObject = std::make_shared<GameObject>();
    const auto gameObject2 = std::make_shared<GameObject>();

    ASSERT_EQ(component->getGameObject(), nullptr);
    component->setGameObject(gameObject);
    ASSERT_EQ(gameObject->getComponents().size(), 1);
    ASSERT_EQ(component->getGameObject(), gameObject);

    component->setGameObject(gameObject2);
    ASSERT_EQ(gameObject->getComponents().size(), 0);
    ASSERT_EQ(gameObject2->getComponents().size(), 1);
    ASSERT_EQ(component->getGameObject(), gameObject2);
}
}  // namespace spark