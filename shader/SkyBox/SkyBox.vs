#version 420 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texcoord;
layout (location = 3) in vec3 color;

out vec3 Texcoord;
out vec3 Normal;
out vec3 Color;

layout(std140, binding=0) uniform TransformMatrix
{
    mat4 project;
    mat4 view;
};
// uniform mat4 viewMatrix;
// uniform mat4 projectMatrix;

void main(){
	Texcoord = normalize(position);
	Normal = normal;
	Color = color;
	mat4 viewMat = mat4(mat3(view));
	vec4 pos = project * viewMat * vec4(position,1.0f);
	gl_Position = pos.xyww;
}