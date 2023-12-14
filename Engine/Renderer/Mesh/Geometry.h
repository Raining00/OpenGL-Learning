#pragma once

#include "Mesh.h"
#define M_PI 3.1415926535
#define M_2PI 6.2831853071
namespace Renderer
{

    class Triangle : public Mesh
    {
        Triangle(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3);

        virtual ~Triangle() = default;
    };

    class Plane : public Mesh
    {
    public:
        Plane(float width, float height, float scale=1.0f);

        virtual ~Plane() = default;
    };

    class Cube : public Mesh
    {
    public:
        Cube(float width, float height, float depth, float scale = 1.0);

        virtual ~Cube() = default;
    };

    class Container : public Mesh
    {
    public:
        Container(float width, float height, float depth);

        virtual ~Container() = default;
    };

    class Sphere : public Mesh
    {
    public:
        Sphere(float radius, int numRings, int numSegments);

        virtual ~Sphere() = default;
    };

    class ScreenQuad : public Mesh
    {
    public:
        ScreenQuad();

        virtual ~ScreenQuad() = default;
    };
} // namespace Renderer
