#include "pch.h"

#include <deque>
#include <string>

#include "Observable.hpp"
#include "IObserver.hpp"

class Subject : public Observable<Subject>
{
    public:
    const std::string& getName() const
    {
        return name;
    };
    void setName(const std::string& _name)
    {
        name = _name;
        notify(this);
    }

    private:
    std::string name{};
};

class Observer : public IObserver<Subject>
{
    public:
    std::string name{};
    // Inherited via IObserver
    void update(const Subject* const observableObject) override
    {
        name = observableObject->getName();
    }
};

TEST(ObserverPatternTest, AddObserverAndNotify)
{
    Subject s;
    const auto o = std::make_shared<Observer>();
    s.add(o);

    ASSERT_EQ(s.getName(), o->name);
    s.setName("newName");
    ASSERT_EQ(s.getName(), o->name);
}

TEST(ObserverPatternTest, RemoveObserverAndNotify)
{
    Subject s;
    const auto o = std::make_shared<Observer>();
    s.add(o);

    ASSERT_EQ(s.getName(), o->name);
    s.setName("newName");
    ASSERT_EQ(s.getName(), o->name);
    s.remove(o);
    s.setName("template");
    ASSERT_NE(s.getName(), o->name);
}

TEST(ObserverPatternTest, AddMultipleObserversAndNotify)
{
    Subject s;
    std::deque<std::shared_ptr<Observer>> observers;
    const uint32_t numberOfObservers{10};

    for(int i = 0; i < numberOfObservers; ++i)
    {
        const auto o = std::make_shared<Observer>();
        observers.push_back(o);
        s.add(o);
    }

    s.setName("newName");

    for(auto& observerPtr : observers)
    {
        ASSERT_EQ(s.getName(), observerPtr->name);
    }
}

TEST(ObserverPatternTest, ExpiredObserverIsRemovedFromList)
{
    Subject s;

    {
        const auto o = std::make_shared<Observer>();
        s.add(o);
        ASSERT_EQ(s.observersCount(), 1);
    }

    s.setName("newName");
    ASSERT_EQ(s.observersCount(), 0);
}

TEST(ObserverPatternTest, NullptrShouldNotBeAdded)
{
    Subject s;
    ASSERT_EQ(s.observersCount(), 0);
    s.add(nullptr);
    ASSERT_EQ(s.observersCount(), 0);
}

TEST(ObserverPatternTest, RemovingNullptrShouldRemoveOneExpiredObserver)
{
    Subject s;
    {
        const auto o = std::make_shared<Observer>();
        s.add(o);
        ASSERT_EQ(s.observersCount(), 1);
        const auto o2 = std::make_shared<Observer>();
        s.add(o2);
        ASSERT_EQ(s.observersCount(), 2);
    }

    s.remove(nullptr);
    ASSERT_EQ(s.observersCount(), 1);
}