#version 420 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;

layout(std140, binding=0) uniform TransformMatrix
{
    mat4 project;
    mat4 view;
};

uniform mat4 model;
// uniform mat4 view;
// uniform mat4 projection;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;  
    
    gl_Position = project * view * vec4(FragPos, 1.0);
}