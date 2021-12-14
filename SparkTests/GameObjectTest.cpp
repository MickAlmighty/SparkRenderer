#include "pch.h"

#include "Component.h"
#include "GameObject.h"
#include "OpenGLContext.hpp"
#include "Scene.h"

class TestComponent1 : public spark::Component
{
    public:
    TestComponent1() : spark::Component() {}
    void update() override {}
};

class TestComponent2 : public spark::Component
{
    public:
    TestComponent2() : spark::Component() {}
    void update() override {}
};

class GameObjectTest : public ::testing::Test
{
    public:
    std::shared_ptr<spark::Scene> scene{nullptr};
    std::shared_ptr<spark::GameObject> sut{nullptr};

    private:
    spark::OpenGLContext oglContext{1280, 720, true, true};

    void SetUp() override
    {
        scene = std::make_shared<spark::Scene>();
        sut = scene->spawnGameObject();
    }
};

namespace spark
{
TEST_F(GameObjectTest, AddingComponentByNullptrTypeNameShouldDoNothing)
{
    ASSERT_EQ(sut->getAllComponentsOfType<Component>().size(), 0);
    sut->addComponent(nullptr);
    ASSERT_EQ(sut->getAllComponentsOfType<Component>().size(), 0);
}

TEST_F(GameObjectTest, AddComponentByType)
{
    const auto component = sut->addComponent<TestComponent1>();
    ASSERT_NE(component, nullptr);
    ASSERT_EQ(component->getGameObject(), sut);
    ASSERT_EQ(sut->getComponent<TestComponent1>(), component);
}

TEST_F(GameObjectTest, AddingTwoComponentsAndTakingOneByTheTypeShouldReturnComponent)
{
    sut->addComponent<TestComponent1>();
    sut->addComponent<TestComponent2>();

    ASSERT_NE(sut->getComponent<TestComponent2>(), nullptr);
}

TEST_F(GameObjectTest, AddingTwoComponentsOfTheSameTypeAndTakingAllOfThemByTheTypeShouldReturnVectorWithBoth)
{
    sut->addComponent<TestComponent2>();
    sut->addComponent<TestComponent2>();

    ASSERT_EQ(sut->getAllComponentsOfType<TestComponent2>().size(), 2);
}

TEST_F(GameObjectTest, TakingComponentByTheTypeWhenThereIsNoComponentsShouldReturnNullptr)
{
    ASSERT_EQ(sut->getComponent<TestComponent2>(), nullptr);
}

TEST_F(GameObjectTest, TakingComponentWithMissmathedTypeShouldReturnNullptr)
{
    sut->addComponent<TestComponent1>();
    ASSERT_EQ(sut->getComponent<TestComponent2>(), nullptr);
}

TEST_F(GameObjectTest, RemovingComponentByNameShouldRemoveComponent)
{
    const auto component = sut->addComponent<TestComponent1>();

    ASSERT_TRUE(sut->removeComponent(component->getName()));
    ASSERT_EQ(sut->getComponent<TestComponent1>(), nullptr);
}

TEST_F(GameObjectTest, RemovedComponentByNameShouldHaveNullptrGameObject)
{
    const auto component = sut->addComponent<TestComponent1>();

    ASSERT_TRUE(sut->removeComponent(component->getName()));
    ASSERT_EQ(component->getGameObject(), nullptr);
}

TEST_F(GameObjectTest, RemovingExactComponentShouldRemoveIt)
{
    const auto component = sut->addComponent<TestComponent1>();

    ASSERT_TRUE(sut->removeComponent(component));
    ASSERT_EQ(sut->getComponent<TestComponent1>(), nullptr);
}

TEST_F(GameObjectTest, RemovedExactComponentShouldHaveNullptrGameObject)
{
    const auto component = sut->addComponent<TestComponent1>();

    ASSERT_TRUE(sut->removeComponent(component));
    ASSERT_EQ(component->getGameObject(), nullptr);
}

TEST_F(GameObjectTest, RemovedComponentOfTypeShouldHaveNullptrGameObject)
{
    const auto component = sut->addComponent<TestComponent1>();

    ASSERT_TRUE(sut->removeComponent<TestComponent1>());
    ASSERT_EQ(component->getGameObject(), nullptr);
}

TEST_F(GameObjectTest, RemovingAllComponentsByTypeShouldRemoveItAll)
{
    sut->addComponent<TestComponent1>();
    sut->addComponent<TestComponent1>();
    const auto component = sut->addComponent<TestComponent2>();

    ASSERT_EQ(sut->getAllComponentsOfType<TestComponent1>().size(), 2);
    sut->removeAllComponentsOfType<TestComponent1>();
    ASSERT_EQ(sut->getAllComponentsOfType<TestComponent1>().size(), 0);
    ASSERT_EQ(sut->getComponent<TestComponent2>(), component);
}

TEST_F(GameObjectTest, AllComponentsRemovedByTypeShouldRemoveItAllShouldHaveNullptrGameObject)
{
    const auto component = sut->addComponent<TestComponent1>();
    const auto component2 = sut->addComponent<TestComponent1>();
    const auto component3 = sut->addComponent<TestComponent2>();

    ASSERT_EQ(sut->getAllComponentsOfType<TestComponent1>().size(), 2);
    sut->removeAllComponentsOfType<TestComponent1>();
    ASSERT_EQ(sut->getAllComponentsOfType<TestComponent1>().size(), 0);
    ASSERT_EQ(sut->getComponent<TestComponent2>(), component3);
    ASSERT_EQ(component->getGameObject(), nullptr);
    ASSERT_EQ(component2->getGameObject(), nullptr);
    ASSERT_EQ(component3->getGameObject(), sut);
}

TEST_F(GameObjectTest, GameObjectSpawnedInSceneShouldHaveParent)
{
    ASSERT_NE(sut->getParent(), nullptr);
}

TEST_F(GameObjectTest, GameObjectCantBeTheParentOfItself)
{
    const auto parent = sut->getParent();
    sut->setParent(sut);
    ASSERT_EQ(sut->getParent(), parent);
    ASSERT_NE(sut->getParent(), sut);
}

TEST_F(GameObjectTest, ChangingParentShouldRemoveGameObjectFromItsChildrenAndAddItToTheNextParentChildrensList)
{
    const auto parent = sut->getParent();

    ASSERT_EQ(parent->getChildren().size(), 1);
    ASSERT_EQ(parent->getChildren()[0], sut);
    ASSERT_EQ(sut->getParent(), parent);

    const std::shared_ptr<GameObject> parent2 = scene->spawnGameObject();
    ASSERT_EQ(parent->getChildren().size(), 2);
    ASSERT_EQ(parent2->getChildren().size(), 0);
    sut->setParent(parent2);
    ASSERT_EQ(parent->getChildren().size(), 1);
    ASSERT_EQ(parent2->getChildren().size(), 1);
    ASSERT_EQ(parent2->getChildren()[0], sut);
    ASSERT_EQ(sut->getParent(), parent2);
}

TEST_F(GameObjectTest, ChangingParentToNullptrShouldNotChangeIt)
{
    const auto parent = sut->getParent();
    ASSERT_EQ(parent->getChildren().size(), 1);
    ASSERT_EQ(sut->getParent(), parent);
    sut->setParent(nullptr);
    ASSERT_EQ(parent->getChildren().size(), 1);
    ASSERT_EQ(sut->getParent(), parent);
}

TEST_F(GameObjectTest, ChangingParentToTheSameParentShouldNotChangeIt)
{
    const auto parent = sut->getParent();

    ASSERT_EQ(parent->getChildren().size(), 1);
    sut->setParent(parent);
    ASSERT_EQ(parent->getChildren().size(), 1);
    ASSERT_EQ(sut->getParent(), parent);
}

TEST_F(GameObjectTest, AddingNullptrAsAChildShouldDoNothing)
{
    ASSERT_EQ(sut->getChildren().size(), 0);
    sut->addChild(nullptr);
    const auto& children = sut->getChildren();
    ASSERT_EQ(children.size(), 0);
}

TEST_F(GameObjectTest, AddingChildShouldIncreaseChildrenCountByOne)
{
    const auto child = scene->spawnGameObject();

    ASSERT_EQ(sut->getChildren().size(), 0);
    sut->addChild(child);
    const auto& children = sut->getChildren();
    ASSERT_EQ(children.size(), 1);
    ASSERT_EQ(children[0], child);
}

TEST_F(GameObjectTest, AddingChildTwiceShouldAddChildOnlyOnce)
{
    const auto child = scene->spawnGameObject();

    ASSERT_EQ(sut->getChildren().size(), 0);
    sut->addChild(child);
    sut->addChild(child);
    ASSERT_EQ(sut->getChildren().size(), 1);
}

TEST_F(GameObjectTest, AddingGameObjectAsChildShouldRemoveItFromPreviousParentAndAddToNewParrent)
{
    const auto parent = sut->getParent();
    ASSERT_EQ(parent->getChildren().size(), 1);
    ASSERT_EQ(sut->getChildren().size(), 0);
    const auto child = sut->getScene()->spawnGameObject();
    ASSERT_EQ(child->getParent(), parent);
    ASSERT_EQ(parent->getChildren().size(), 2);
    sut->addChild(child);
    ASSERT_EQ(parent->getChildren().size(), 1);
    ASSERT_EQ(sut->getChildren().size(), 1);
    ASSERT_EQ(child->getParent(), sut);
}

TEST_F(GameObjectTest, ChildRemoval)
{
    const std::shared_ptr<GameObject> child = scene->spawnGameObject();

    ASSERT_EQ(sut->getChildren().size(), 0);
    sut->addChild(child);
    const auto& children = sut->getChildren();

    ASSERT_TRUE(sut->removeChild(child));
    ASSERT_EQ(children.size(), 0);
    ASSERT_EQ(child->getParent(), nullptr);
}

TEST_F(GameObjectTest, NullptrChildRemoval)
{
    const std::shared_ptr<GameObject> child = scene->spawnGameObject();

    ASSERT_EQ(sut->getChildren().size(), 0);
    sut->addChild(child);
    const auto& children = sut->getChildren();
    ASSERT_EQ(children.size(), 1);

    ASSERT_FALSE(sut->removeChild(nullptr));
    ASSERT_EQ(children.size(), 1);
}
}  // namespace spark