#version 420 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

layout(std140, binding=0) uniform TransformMatrix
{
    mat4 project;
    mat4 view;
};

out VS_OUT
{
    vec3 Normal;
    mat4 projectMatrix;
}vs_out;

uniform mat4 modelMatrix;
uniform mat3 normalMatrix;

void main()
{
    vs_out.Normal = normalize(normalMatrix * aNormal);
    vs_out.projectMatrix = project;
    gl_Position = view * modelMatrix * vec4(aPos, 1.0);
}