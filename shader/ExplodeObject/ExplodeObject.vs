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
    vec3 originalPos;
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectMatrix;
}vs_out;

uniform bool instance;

uniform mat4 modelMatrix;
uniform mat3 normalMatrix;

void main()
{
    vs_out.FragPos = vec3(modelMatrix * vec4(aPos, 1.0));
    vs_out.Normal = normalMatrix * aNormal; 
    vs_out.TexCoords = aTexCoords;
    vs_out.originalPos = aPos;
    vs_out.modelMatrix = modelMatrix;
    vs_out.viewMatrix = view;
    vs_out.projectMatrix = project;
    gl_Position = project * view * vec4(vs_out.FragPos, 1.0);
}