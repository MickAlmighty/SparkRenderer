#ifndef HID_H
#define HID_H

#include <map>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

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
    static void clearStates();
    static bool isKeyPressed(int key);
    static bool isKeyReleased(int key);
    static bool isKeyHeld(int key);

    private:
    static std::map<int, int> keyStates;

    HID() = default;
    ~HID() = default;
};

}  // namespace spark
#endif