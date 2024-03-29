#version 420 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec4 aColor;

out VS_OUT
{
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
    vec3 eyeSpacePos;
    vec4 FragPosLightSpace;
    vec4 Color;
    mat4 projectMatrix;
}vs_out;

layout(std140, binding=0) uniform TransformMatrix
{
    mat4 project;
    mat4 view;
};

uniform mat4 modelMatrix;
uniform mat3 normalMatrix;
uniform mat4 lightSpaceMatrix;
uniform bool instance;
uniform float pointScale;
uniform float pointSize;

void main()
{
    vec3 FragPos;
    vs_out.Normal = normalMatrix * aNormal; 
    vs_out.TexCoords = aTexCoords;
    FragPos = vec3(modelMatrix * vec4(aPos, 1.0));
    vs_out.FragPosLightSpace = lightSpaceMatrix * vec4(FragPos, 1.0);
    vs_out.FragPos = FragPos;
    vs_out.eyeSpacePos = (view * vec4(FragPos, 1.0)).xyz;
    gl_PointSize = -pointScale * pointSize / vs_out.eyeSpacePos.z;
    gl_Position = project * view * vec4(FragPos, 1.0);
}