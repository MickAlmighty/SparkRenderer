#include "HID.h"

#include "Spark.h"
#include "Logging.h"

namespace spark
{
Mouse HID::mouse{};

void HID::scroll_callback(double xoffset, double yoffset)
{
    if(yoffset < 0.0)
        mouse.scroll = ScrollStatus::NEGATIVE;
    else if(yoffset > 0.0)
        mouse.scroll = ScrollStatus::POSITIVE;
}

void HID::key_callback(int key, int scancode, int action, int mods)
{
    processKeys(key, action);
}

void HID::mouse_button_callback(int button, int action, int mods)
{
    processKeys(button, action);
}

void HID::cursor_position_callback(double xpos, double ypos)
{
    static double lastXpos = static_cast<double>(Spark::get().getRenderingContext().width) * 0.5,
                  lastYPos = static_cast<double>(Spark::get().getRenderingContext().height) * 0.5;

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

void HID::updateStates()
{
    mouse.direction *= 0;
    mouse.scroll = ScrollStatus::IDLE;

    for(auto& [key, state] : keyStates)
    {
        if(state == State::PRESSED)
        {
            bool& processedOnce = keyPressedOnce[key];
            if(!processedOnce)
            {
                processedOnce = true;
            }
        }
        if(state == State::RELEASED)
        {
            state = State::NONE;
            // SPARK_INFO("State of key {} was cleared.", static_cast<uint32_t>(key));

            keyPressedOnce[key] = false;
        }
    }
}

bool HID::isKeyPressed(Key key)
{
    return keyStates[key] == State::PRESSED && keyPressedOnce[key] == false;
}

bool HID::isKeyReleased(Key key)
{
    return keyStates[key] == State::RELEASED;
}

State HID::getKeyState(Key key)
{
    return keyStates[key];
}

bool HID::isKeyDown(Key key)
{
    return keyStates[key] == State::DOWN;
}

bool HID::isKeyPressedOrDown(Key key)
{
    const State& state = keyStates[key];
    return state == State::PRESSED || state == State::DOWN;
}

Key HID::getMappedKey(int glfwKey)
{
    const auto keyIterator = glfwKeyMapping.find(glfwKey);
    if(keyIterator != glfwKeyMapping.end())
    {
        return keyIterator->second;
    }
    return Key::UNKNOWN_KEY;
}

void HID::processKeys(int key, int action)
{
    const Key mappedKey = getMappedKey(key);

    if(mappedKey == Key::UNKNOWN_KEY)
        return;

    // SPARK_INFO("Key callback invoked for key {}, with state {}", static_cast<int32_t>(mappedKey), action);

    if(action == GLFW_PRESS)
    {
        keyStates[mappedKey] = State::PRESSED;
    }
    else if(action == GLFW_RELEASE)
    {
        keyStates[mappedKey] = State::RELEASED;
    }
    else if(action == GLFW_REPEAT)
    {
        keyStates[mappedKey] = State::DOWN;
    }
}

const std::map<int, spark::Key> HID::glfwKeyMapping = {
    {GLFW_KEY_ESCAPE, Key::ESC},
    {GLFW_KEY_F1, Key::F1},
    {GLFW_KEY_F2, Key::F2},
    {GLFW_KEY_F3, Key::F3},
    {GLFW_KEY_F4, Key::F4},
    {GLFW_KEY_F5, Key::F5},
    {GLFW_KEY_F6, Key::F6},
    {GLFW_KEY_F7, Key::F7},
    {GLFW_KEY_F8, Key::F8},
    {GLFW_KEY_F9, Key::F9},
    {GLFW_KEY_F10, Key::F10},
    {GLFW_KEY_1, Key::NUM_1},
    {GLFW_KEY_2, Key::NUM_2},
    {GLFW_KEY_3, Key::NUM_3},
    {GLFW_KEY_4, Key::NUM_4},
    {GLFW_KEY_5, Key::NUM_5},
    {GLFW_KEY_6, Key::NUM_6},
    {GLFW_KEY_7, Key::NUM_7},
    {GLFW_KEY_8, Key::NUM_8},
    {GLFW_KEY_9, Key::NUM_9},
    {GLFW_KEY_BACKSPACE, Key::BACK_SPACE},
    {GLFW_KEY_TAB, Key::TAB},
    {GLFW_KEY_Q, Key::Q},
    {GLFW_KEY_W, Key::W},
    {GLFW_KEY_E, Key::E},
    {GLFW_KEY_R, Key::R},
    {GLFW_KEY_T, Key::T},
    {GLFW_KEY_Y, Key::Y},
    {GLFW_KEY_U, Key::U},
    {GLFW_KEY_I, Key::I},
    {GLFW_KEY_O, Key::O},
    {GLFW_KEY_P, Key::P},
    {GLFW_KEY_A, Key::A},
    {GLFW_KEY_S, Key::S},
    {GLFW_KEY_D, Key::D},
    {GLFW_KEY_F, Key::F},
    {GLFW_KEY_G, Key::G},
    {GLFW_KEY_H, Key::H},
    {GLFW_KEY_J, Key::J},
    {GLFW_KEY_K, Key::K},
    {GLFW_KEY_L, Key::L},
    {GLFW_KEY_ENTER, Key::ENTER},
    {GLFW_KEY_LEFT_SHIFT, Key::LEFT_SHIFT},
    {GLFW_KEY_Z, Key::Z},
    {GLFW_KEY_X, Key::X},
    {GLFW_KEY_C, Key::C},
    {GLFW_KEY_V, Key::V},
    {GLFW_KEY_B, Key::B},
    {GLFW_KEY_N, Key::N},
    {GLFW_KEY_M, Key::M},
    {GLFW_KEY_RIGHT_SHIFT, Key::RIGHT_SHIFT},
    {GLFW_KEY_LEFT_CONTROL, Key::LEFT_CTRL},
    {GLFW_KEY_LEFT_ALT, Key::LEFT_ALT},
    {GLFW_KEY_SPACE, Key::SPACE_BAR},
    {GLFW_KEY_RIGHT_ALT, Key::RIGHT_ALT},
    {GLFW_KEY_RIGHT_CONTROL, Key::RIGHT_CTRL},
    {GLFW_KEY_LEFT, Key::ARROW_LEFT},
    {GLFW_KEY_DOWN, Key::ARROW_DOWN},
    {GLFW_KEY_RIGHT, Key::ARROW_RIGHT},
    {GLFW_KEY_UP, Key::ARROW_UP},
    {GLFW_MOUSE_BUTTON_LEFT, Key::MOUSE_LEFT},
    {GLFW_MOUSE_BUTTON_MIDDLE, Key::MOUSE_MIDDLE},
    {GLFW_MOUSE_BUTTON_RIGHT, Key::MOUSE_RIGHT},
};

}  // namespace spark
