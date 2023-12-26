#version 420 core
layout (location = 0) in vec4 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out VS_OUT
{
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
    vec3 eyeSpacePos;
    vec4 FragPosLightSpace;
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
    vec3 position = vec3(aPos);
    vs_out.Normal = normalMatrix * aNormal; 
    vs_out.TexCoords = aTexCoords;
    FragPos = vec3(modelMatrix * vec4(position, 1.0));
    vs_out.FragPosLightSpace = lightSpaceMatrix * vec4(FragPos, 1.0);
    vs_out.FragPos = FragPos;
    vs_out.eyeSpacePos = (view * vec4(FragPos, 1.0)).xyz;
    gl_PointSize = -pointScale * pointSize / vs_out.eyeSpacePos.z;
    // gl_PointSize = max(1.0, -pointScale * pointSize / (1.0 - vs_out.eyeSpacePos.z));
    gl_Position = project * view * vec4(FragPos, 1.0);
}