#version 420 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoords;
layout(std140, binding=0) uniform TransformMatrix
{
    mat4 project;
    mat4 view;
};
uniform bool instance;
uniform mat4 modelMatrix;
// uniform mat4 viewMatrix;
// uniform mat4 projectMatrix;
uniform mat3 normalMatrix;

void main()
{
    FragPos = vec3(modelMatrix * vec4(aPos, 1.0));
    Normal = normalMatrix * aNormal; 
    TexCoords = aTexCoords;
    gl_Position = project * view * vec4(FragPos, 1.0);
}