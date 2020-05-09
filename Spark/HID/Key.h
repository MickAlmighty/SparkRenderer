#ifndef KEY_H
#define KEY_H

namespace spark
{

enum class Key
{
    UNKNOWN_KEY = 0, ESC, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10,
    NUM_1, NUM_2, NUM_3, NUM_4, NUM_5, NUM_6, NUM_7, NUM_8, NUM_9, BACK_SPACE,
    TAB, Q, W, E, R, T, Y, U, I, O, P,
    A, S, D, F, G, H, J, K, L, ENTER,
    LEFT_SHIFT, Z, X, C, V, B, N, M, RIGHT_SHIFT,
    LEFT_CTRL, LEFT_ALT, SPACE_BAR, RIGHT_ALT, RIGHT_CTRL,
    ARROW_LEFT, ARROW_DOWN, ARROW_RIGHT, ARROW_UP
};

}

#endif