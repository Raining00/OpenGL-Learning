#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectMatrix;
uniform mat3 normalMatrix;

out vec3 Normal;
out vec3 FragPos;
out vec2 TexCoords;

void main()
{
    Normal = mat3(transpose(inverse(modelMatrix))) * aNormal;
    gl_Position = projectMatrix * viewMatrix * modelMatrix * vec4(aPos, 1.0) + vec4(0.1 * Normal, 0.0);
}