#pragma once

#include <vector>

#include "glad_glfw3.h"
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"

class ScreenQuad final
{
    public:
    ScreenQuad();
    ScreenQuad(const ScreenQuad&) = delete;
    ScreenQuad(ScreenQuad&&) = delete;
    ScreenQuad& operator=(const ScreenQuad&) = delete;
    ScreenQuad& operator=(ScreenQuad&&) = delete;
    ~ScreenQuad();

    void draw() const;

    private:
    struct QuadVertex final
    {
        glm::vec3 pos;
        glm::vec2 texCoords;
    };

    inline static size_t counter{0};
    inline static GLuint vao{0};
    inline static GLuint vbo{0};

    inline static const std::vector<QuadVertex> vertices{
        {{-1.0f, 1.0f, 0.0f}, {0.0f, 1.0f}}, {{1.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},  {{1.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},

        {{-1.0f, 1.0f, 0.0f}, {0.0f, 1.0f}}, {{-1.0f, -1.0f, 0.0f}, {0.0f, 0.0f}}, {{1.0f, -1.0f, 0.0f}, {1.0f, 0.0f}}};
};

inline ScreenQuad::ScreenQuad()
{
    if(counter == 0)
    {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(QuadVertex), &vertices[0], GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(QuadVertex), reinterpret_cast<void*>(offsetof(QuadVertex, pos)));

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(QuadVertex), reinterpret_cast<void*>(offsetof(QuadVertex, texCoords)));

        glBindVertexArray(0);
    }
    ++counter;
}

inline ScreenQuad::~ScreenQuad()
{
    --counter;
    if(counter == 0)
    {
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
    }
}

inline void ScreenQuad::draw() const
{
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size()));
    glBindVertexArray(0);
}
