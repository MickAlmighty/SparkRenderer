#pragma once

#include <map>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "Key.h"
#include "State.h"

namespace spark
{
enum class State;

enum class ScrollStatus
{
    IDLE,
    POSITIVE,
    NEGATIVE
};

struct Mouse
{
    glm::vec2 direction;
    glm::vec2 screenPosition;

    ScrollStatus getScrollStatus() const
    {
        return scroll;
    }

    private:
    ScrollStatus scroll{ScrollStatus::IDLE};
    friend class HID;
};

class HID
{
    public:
    static Mouse mouse;
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
    static void updateStates();
    static bool isKeyPressed(Key key);
    static bool isKeyDown(Key key);
    static bool isKeyPressedOrDown(Key key);
    static bool isKeyReleased(Key key);
    static State getKeyState(Key key);

    private:
    static const std::map<int, Key> glfwKeyMapping;  // defined in HID.cpp
    inline static std::map<Key, State> keyStates{};
    inline static std::map<Key, bool> keyPressedOnce{};

    HID() = default;
    ~HID() = default;

    static Key getMappedKey(int glfwKey);
    static void processKeys(int key, int action);
};

}  // namespace spark