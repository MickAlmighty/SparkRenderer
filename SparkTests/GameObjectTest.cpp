#include "pch.h"

#include "Component.h"
#include "GameObject.h"

class TestComponent1 : public spark::Component
{
    public:
    TestComponent1(std::string&& name = "TestComponent1") : spark::Component(std::move(name)) {}
    void update() override{};
    void fixedUpdate() override{};
    void drawGUI() override{};
};

class TestComponent2 : public spark::Component
{
    public:
    TestComponent2(std::string&& name = "TestComponent2") : spark::Component(std::move(name)) {}
    void update() override{};
    void fixedUpdate() override{};
    void drawGUI() override{};
};

namespace spark
{
TEST(GameObjectTest, AddingNullptrComponentShouldDoNothing)
{
    std::shared_ptr<GameObject> gameObject = std::make_shared<GameObject>();

    ASSERT_EQ(gameObject->getAllComponentsOfType<Component>().size(), 0);
    gameObject->addComponent(nullptr);
    ASSERT_EQ(gameObject->getAllComponentsOfType<Component>().size(), 0);
}

TEST(GameObjectTest, AddingTheSameComponentTwiceShouldAddItOnlyOnce)
{
    std::shared_ptr<GameObject> gameObject = std::make_shared<GameObject>();
    const auto component = std::make_shared<TestComponent1>();

    gameObject->addComponent(component);
    gameObject->addComponent(component);

    ASSERT_EQ(gameObject->getAllComponentsOfType<Component>().size(), 1);
}

TEST(GameObjectTest, AddingComponentAndTakingItByTheTypeShouldReturnComponent)
{
    std::shared_ptr<GameObject> gameObject = std::make_shared<GameObject>();
    const auto component = std::make_shared<TestComponent1>();

    gameObject->addComponent(component);

    ASSERT_NE(gameObject->getComponent<TestComponent1>(), nullptr);
}

TEST(GameObjectTest, AddingTwoComponentsAndTakingOneByTheTypeShouldReturnComponent)
{
    std::shared_ptr<GameObject> gameObject = std::make_shared<GameObject>();
    const auto component = std::make_shared<TestComponent1>();
    const auto component2 = std::make_shared<TestComponent2>();

    gameObject->addComponent(component);
    gameObject->addComponent(component2);

    ASSERT_NE(gameObject->getComponent<TestComponent2>(), nullptr);
}

TEST(GameObjectTest, AddingTwoComponentsOfTheSameTypeAndTakingAllOfThemByTheTypeShouldReturnVectorWithBoth)
{
    std::shared_ptr<GameObject> gameObject = std::make_shared<GameObject>();
    const auto component = std::make_shared<TestComponent2>();
    const auto component2 = std::make_shared<TestComponent2>();

    gameObject->addComponent(component);
    gameObject->addComponent(component2);

    ASSERT_EQ(gameObject->getAllComponentsOfType<TestComponent2>().size(), 2);
}

TEST(GameObjectTest, TakingComponentByTheTypeWhenThereIsNoComponentsShouldReturnNullptr)
{
    std::shared_ptr<GameObject> gameObject = std::make_shared<GameObject>();

    ASSERT_EQ(gameObject->getComponent<TestComponent2>(), nullptr);
}

TEST(GameObjectTest, TakingComponentWithMissmathedTypeShouldReturnNullptr)
{
    std::shared_ptr<GameObject> gameObject = std::make_shared<GameObject>();
    const auto component = std::make_shared<TestComponent1>();
    gameObject->addComponent(component);

    ASSERT_EQ(gameObject->getComponent<TestComponent2>(), nullptr);
}

TEST(GameObjectTest, RemovingComponentByNameShouldRemoveComponent)
{
    std::shared_ptr<GameObject> gameObject = std::make_shared<GameObject>();
    const auto component = std::make_shared<TestComponent1>();
    gameObject->addComponent(component);

    ASSERT_TRUE(gameObject->removeComponent(component->getName()));
    ASSERT_EQ(gameObject->getComponent<TestComponent1>(), nullptr);
}

TEST(GameObjectTest, RemovedComponentByNameShouldHaveNullptrGameObject)
{
    std::shared_ptr<GameObject> gameObject = std::make_shared<GameObject>();
    const auto component = std::make_shared<TestComponent1>();
    gameObject->addComponent(component);

    ASSERT_TRUE(gameObject->removeComponent(component->getName()));
    ASSERT_EQ(component->getGameObject(), nullptr);
}

TEST(GameObjectTest, RemovingExactComponentShouldRemoveIt)
{
    std::shared_ptr<GameObject> gameObject = std::make_shared<GameObject>();
    const auto component = std::make_shared<TestComponent1>();
    gameObject->addComponent(component);

    ASSERT_TRUE(gameObject->removeComponent(component));
    ASSERT_EQ(gameObject->getComponent<TestComponent1>(), nullptr);
}

TEST(GameObjectTest, RemovedExactComponentShouldHaveNullptrGameObject)
{
    std::shared_ptr<GameObject> gameObject = std::make_shared<GameObject>();
    const auto component = std::make_shared<TestComponent1>();
    gameObject->addComponent(component);

    ASSERT_TRUE(gameObject->removeComponent(component));
    ASSERT_EQ(component->getGameObject(), nullptr);
}

TEST(GameObjectTest, RemovedComponentOfTypeShouldHaveNullptrGameObject)
{
    std::shared_ptr<GameObject> gameObject = std::make_shared<GameObject>();
    const auto component = std::make_shared<TestComponent1>();
    gameObject->addComponent(component);

    ASSERT_TRUE(gameObject->removeComponent<TestComponent1>());
    ASSERT_EQ(component->getGameObject(), nullptr);
}

TEST(GameObjectTest, RemovingAllComponentsByTypeShouldRemoveItAll)
{
    std::shared_ptr<GameObject> gameObject = std::make_shared<GameObject>();
    const auto component = std::make_shared<TestComponent1>();
    const auto component2 = std::make_shared<TestComponent1>();
    const auto component3 = std::make_shared<TestComponent2>();
    gameObject->addComponent(component);
    gameObject->addComponent(component2);
    gameObject->addComponent(component3);

    ASSERT_EQ(gameObject->getAllComponentsOfType<TestComponent1>().size(), 2);
    gameObject->removeAllComponentsOfType<TestComponent1>();
    ASSERT_EQ(gameObject->getAllComponentsOfType<TestComponent1>().size(), 0);
    ASSERT_EQ(gameObject->getComponent<TestComponent2>(), component3);
}

TEST(GameObjectTest, AllComponentsRemovedByTypeShouldRemoveItAllShouldHaveNullptrGameObject)
{
    std::shared_ptr<GameObject> gameObject = std::make_shared<GameObject>();
    const auto component = std::make_shared<TestComponent1>();
    const auto component2 = std::make_shared<TestComponent1>();
    const auto component3 = std::make_shared<TestComponent2>();
    gameObject->addComponent(component);
    gameObject->addComponent(component2);
    gameObject->addComponent(component3);

    ASSERT_EQ(gameObject->getAllComponentsOfType<TestComponent1>().size(), 2);
    gameObject->removeAllComponentsOfType<TestComponent1>();
    ASSERT_EQ(gameObject->getAllComponentsOfType<TestComponent1>().size(), 0);
    ASSERT_EQ(gameObject->getComponent<TestComponent2>(), component3);
    ASSERT_EQ(component->getGameObject(), nullptr);
    ASSERT_EQ(component2->getGameObject(), nullptr);
    ASSERT_EQ(component3->getGameObject(), gameObject);
}

TEST(GameObjectTest, GettingParentOfGameObjectWhenGameObjectHasNoParentShouldReturnNullptr)
{
    const std::shared_ptr<GameObject> gameObject = std::make_shared<GameObject>();
    ASSERT_EQ(gameObject->getParent(), nullptr);
}

TEST(GameObjectTest, GettingParentOfGameObjectWhenGameObjectHasParentShouldReturnParent)
{
    std::shared_ptr<GameObject> gameObject = std::make_shared<GameObject>();
    const std::shared_ptr<GameObject> parent = std::make_shared<GameObject>();

    gameObject->setParent(parent);

    ASSERT_EQ(gameObject->getParent(), parent);
}

TEST(GameObjectTest, ChangingParentShouldRemoveGameObjectFromItsChildrenAndAddItToTheNextParentChildrensList)
{
    std::shared_ptr<GameObject> gameObject = std::make_shared<GameObject>();
    const std::shared_ptr<GameObject> parent = std::make_shared<GameObject>();

    ASSERT_EQ(parent->getChildren().size(), 0);
    gameObject->setParent(parent);
    ASSERT_EQ(parent->getChildren().size(), 1);
    ASSERT_EQ(parent->getChildren()[0], gameObject);
    ASSERT_EQ(gameObject->getParent(), parent);

    const std::shared_ptr<GameObject> parent2 = std::make_shared<GameObject>();
    gameObject->setParent(parent2);
    ASSERT_EQ(parent->getChildren().size(), 0);
    ASSERT_EQ(parent2->getChildren().size(), 1);
    ASSERT_EQ(parent2->getChildren()[0], gameObject);
    ASSERT_EQ(gameObject->getParent(), parent2);
}

TEST(GameObjectTest, ChangingParentToNullptrShouldNotChangeIt)
{
    std::shared_ptr<GameObject> gameObject = std::make_shared<GameObject>();
    const std::shared_ptr<GameObject> parent = std::make_shared<GameObject>();

    ASSERT_EQ(parent->getChildren().size(), 0);
    ASSERT_EQ(gameObject->getParent(), nullptr);
    gameObject->setParent(parent);
    ASSERT_EQ(parent->getChildren().size(), 1);
    ASSERT_EQ(gameObject->getParent(), parent);
    gameObject->setParent(nullptr);
    ASSERT_EQ(parent->getChildren().size(), 1);
    ASSERT_EQ(gameObject->getParent(), parent);
}

TEST(GameObjectTest, ChangingParentToTheSameParentShouldNotChangeIt)
{
    std::shared_ptr<GameObject> gameObject = std::make_shared<GameObject>();
    const std::shared_ptr<GameObject> parent = std::make_shared<GameObject>();

    ASSERT_EQ(parent->getChildren().size(), 0);
    gameObject->setParent(parent);
    ASSERT_EQ(parent->getChildren().size(), 1);
    ASSERT_EQ(gameObject->getParent(), parent);
    gameObject->setParent(parent);
    ASSERT_EQ(gameObject->getParent(), parent);
}

TEST(GameObjectTest, AddingNullptrAsAChildShouldDoNothing)
{
    std::shared_ptr<GameObject> gameObject = std::make_shared<GameObject>();

    ASSERT_EQ(gameObject->getChildren().size(), 0);
    gameObject->addChild(nullptr);
    const auto& children = gameObject->getChildren();
    ASSERT_EQ(children.size(), 0);
}

TEST(GameObjectTest, AddingChildTwiceShouldAddChildOnlyOnce)
{
    std::shared_ptr<GameObject> gameObject = std::make_shared<GameObject>();
    const std::shared_ptr<GameObject> child = std::make_shared<GameObject>();

    ASSERT_EQ(gameObject->getChildren().size(), 0);
    gameObject->addChild(child);
    gameObject->addChild(child);
    ASSERT_EQ(gameObject->getChildren().size(), 1);
}

TEST(GameObjectTest, AddingChildShouldIncreaseChildrenCountByOne)
{
    std::shared_ptr<GameObject> gameObject = std::make_shared<GameObject>();
    const std::shared_ptr<GameObject> child = std::make_shared<GameObject>();

    ASSERT_EQ(gameObject->getChildren().size(), 0);
    gameObject->addChild(child);
    const auto& children = gameObject->getChildren();
    ASSERT_EQ(children.size(), 1);
    ASSERT_EQ(children[0], child);
}

TEST(GameObjectTest, ChildRemoval)
{
    std::shared_ptr<GameObject> gameObject = std::make_shared<GameObject>();
    const std::shared_ptr<GameObject> child = std::make_shared<GameObject>();

    ASSERT_EQ(gameObject->getChildren().size(), 0);
    gameObject->addChild(child);
    const auto& children = gameObject->getChildren();
    ASSERT_EQ(children.size(), 1);
    ASSERT_EQ(children[0], child);

    ASSERT_TRUE(gameObject->removeChild(child));
    ASSERT_EQ(children.size(), 0);
}

TEST(GameObjectTest, NullptrChildRemoval)
{
    std::shared_ptr<GameObject> gameObject = std::make_shared<GameObject>();
    const std::shared_ptr<GameObject> child = std::make_shared<GameObject>();

    ASSERT_EQ(gameObject->getChildren().size(), 0);
    gameObject->addChild(child);
    const auto& children = gameObject->getChildren();
    ASSERT_EQ(children.size(), 1);

    ASSERT_FALSE(gameObject->removeChild(nullptr));
    ASSERT_EQ(children.size(), 1);
}
}  // namespace spark