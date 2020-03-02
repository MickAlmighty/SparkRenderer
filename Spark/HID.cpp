#include "HID.h"

#include "Spark.h"
#include "Logging.h"

namespace spark
{
Mouse HID::mouse{};
std::map<int, int> HID::keyStates;

void HID::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    static bool mouseDisabled = false;
    if(key == GLFW_KEY_LEFT_ALT && action == GLFW_RELEASE && mouseDisabled)
    {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        mouseDisabled = false;
    }
    else if(key == GLFW_KEY_LEFT_ALT && action == GLFW_RELEASE && !mouseDisabled)
    {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        mouseDisabled = true;
    }

    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    const int state = glfwGetKey(window, key);
    keyStates[key] = state;
}

void HID::cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    static double lastXpos = Spark::WIDTH * 0.5f, lastYPos = Spark::HEIGHT * 0.5f;

    mouse.direction.x = static_cast<float>(xpos - lastXpos);
    mouse.direction.y = static_cast<float>(ypos - lastYPos);
    // mouse.direction *= Clock::getDeltaTime();
    lastXpos = xpos;
    lastYPos = ypos;

    mouse.screenPosition.x = static_cast<float>(xpos);
    mouse.screenPosition.y = static_cast<float>(ypos);

    // SPARK_DEBUG("Mouse X: {}, Mouse Y: {}", xpos, ypos);
    // SPARK_DEBUG("Mouse X_Dir: {}, Mouse Y_Dir: {}", mouse.direction.x, mouse.direction.y);
}

void HID::clearStates()
{
    mouse.direction *= 0;
}

bool HID::isKeyPressed(int key)
{
    const auto key_it = keyStates.find(key);
    if(key_it != keyStates.end())
    {
        return key_it->second == GLFW_PRESS;
    }
    return false;
}

bool HID::isKeyReleased(int key)
{
    const auto key_it = keyStates.find(key);
    if(key_it != keyStates.end())
    {
        return key_it->second == GLFW_RELEASE;
    }
    return false;
}

bool HID::isKeyHeld(int key)
{
    const auto key_it = keyStates.find(key);
    if(key_it != keyStates.end())
    {
        return key_it->second == GLFW_REPEAT;
    }
    return false;
}

}  // namespace spark
