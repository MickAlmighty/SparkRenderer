#pragma once

#include <deque>
#include <memory>

#include "IObserver.hpp"

template<typename ObservableType>
class Observable
{
public:
    void add(const std::shared_ptr<IObserver<ObservableType>>& observer);
    void remove(const std::shared_ptr<IObserver<ObservableType>>& observer);
    void notify(const ObservableType* const o) const;
    virtual ~Observable() = default;
private:
    std::deque<std::shared_ptr<IObserver<ObservableType>>> observers;
};

template<typename ObservableType>
inline void Observable<ObservableType>::add(const std::shared_ptr<IObserver<ObservableType>>& observer)
{
    observers.push_back(observer);
}

template<typename ObservableType>
inline void Observable<ObservableType>::remove(const std::shared_ptr<IObserver<ObservableType>>& observer)
{
    const auto it = std::find(observers.begin(), observers.end(), observer);
    if (it != observers.end())
    {
        observers.erase(it);
    }
}

template<typename ObservableType>
inline void Observable<ObservableType>::notify(const ObservableType* const o) const
{
    for (auto& observer : observers)
    {
        observer->update(o);
    }
}
