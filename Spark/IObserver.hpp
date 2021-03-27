#pragma once

template<typename T>
class IObserver
{
    public:
    virtual void update(const T* const) = 0;
    virtual ~IObserver() = default;
};