#version 330 core

layout (triangles) in;
layout (line_strip, max_vertices = 6) out;

in VS_OUT
{
    vec3 Normal;
    mat4 projectMatrix;
}gs_in[];

uniform float normalDistance;

void GenerateLine(int index)
{
    gl_Position = gs_in[index].projectMatrix * gl_in[index].gl_Position;
    EmitVertex();
    gl_Position = gs_in[index].projectMatrix * (gl_in[index].gl_Position +
                                     vec4(gs_in[index].Normal, 0.0) * normalDistance);
    EmitVertex();
    EndPrimitive();
}

void main()
{
    for(int i = 0; i < gl_in.length(); i++)
        GenerateLine(i);
}