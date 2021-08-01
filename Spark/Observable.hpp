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
    void notify(const ObservableType* const o);
    size_t observersCount() const;
    virtual ~Observable() = default;

    private:
    std::deque<std::weak_ptr<IObserver<ObservableType>>> observers;
};

template<typename ObservableType>
inline void Observable<ObservableType>::add(const std::shared_ptr<IObserver<ObservableType>>& observer)
{
    if(observer != nullptr)
        observers.push_back(observer);
}

template<typename ObservableType>
inline void Observable<ObservableType>::remove(const std::shared_ptr<IObserver<ObservableType>>& observer)
{
    const auto it = std::find_if(observers.cbegin(), observers.cend(),
                                 [&observer](const auto& observer_weak_ptr) { return observer_weak_ptr.lock() == observer; });
    if(it != observers.end())
    {
        observers.erase(it);
    }
}

template<typename ObservableType>
inline void Observable<ObservableType>::notify(const ObservableType* const o)
{
    for(auto it = observers.begin(); it != observers.end();)
    {
        if(it->expired())
        {
            it = observers.erase(it);
        }
        else
        {
            it->lock()->update(o);
            ++it;
        }
    }
}

template<typename ObservableType>
size_t Observable<ObservableType>::observersCount() const
{
    return observers.size();
}
