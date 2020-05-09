#ifndef HID_H
#define HID_H

#include <map>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "Key.h"
#include "State.h"

namespace spark
{

struct Mouse
{
    glm::vec2 direction;
    glm::vec2 screenPosition;
};

class HID
{
    public:
    static Mouse mouse;
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
    static void processInputDevicesStates();
    static bool isKeyPressed(Key key);
    static bool isKeyDown(Key key);
    static bool isKeyPressedOrDown(Key key);
    static bool isKeyReleased(Key key);
    

    private:
    static const std::map<int, Key> glfwKeyMapping; //defined in HID.cpp
    inline static std::map<Key, State> keyStates{};
    inline static std::map<Key, bool> keyPressedOnce{};

    HID() = default;
    ~HID() = default;

    static Key getMappedKey(int glfwKey);
};

}  // namespace spark
#endif