#version 420 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in mat4 instanceMatrix;

out VS_OUT
{
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
    vec4 FragPosLightSpace;
}vs_out;

uniform bool instance;

layout(std140, binding=0) uniform TransformMatrix
{
    mat4 project;
    mat4 view;
};

uniform mat4 modelMatrix;
uniform mat3 normalMatrix;
uniform mat4 lightSpaceMatrix;

void main()
{
    vec3 FragPos;
    if (instance)
    {
        FragPos = vec3(instanceMatrix * vec4(aPos, 1.0));
        vs_out.Normal = mat3(transpose(inverse(instanceMatrix))) * aNormal;
    }
    else
    {
        FragPos = vec3(modelMatrix * vec4(aPos, 1.0));
        vs_out.Normal = normalMatrix * aNormal;
    }
    vs_out.TexCoords = aTexCoords;
    vs_out.FragPosLightSpace = lightSpaceMatrix * vec4(FragPos, 1.0);
    vs_out.FragPos = FragPos;
    gl_Position = project * view * vec4(FragPos, 1.0);
}