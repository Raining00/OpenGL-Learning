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


layout(std140, binding=0) uniform TransformMatrix
{
    mat4 project;
    mat4 view;
};

uniform mat4 modelMatrix;
uniform mat3 normalMatrix;
uniform mat4 lightSpaceMatrix;
uniform bool instance;

void main()
{
    vec3 FragPos;
    vs_out.Normal = normalMatrix * aNormal; 
    vs_out.TexCoords = aTexCoords;
    if(!instance)
        FragPos = vec3(modelMatrix * vec4(aPos, 1.0));
    else
        FragPos = vec3(modelMatrix * instanceMatrix * vec4(aPos, 1.0));

    vs_out.FragPosLightSpace = lightSpaceMatrix * vec4(FragPos, 1.0);
    vs_out.FragPos = FragPos;
    gl_Position = project * view * vec4(FragPos, 1.0);
}