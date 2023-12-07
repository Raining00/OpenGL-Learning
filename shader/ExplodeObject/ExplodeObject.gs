#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in VS_OUT
{
    vec3 originalPos;
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectMatrix;
}gs_in[];

out GS_OUT
{
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
}gs_out;

uniform float sperateDistance;

vec3 GetNormal()
{
    vec3 a = vec3(gs_in[0].originalPos) - vec3(gs_in[1].originalPos);
    vec3 b = vec3(gs_in[2].originalPos) - vec3(gs_in[1].originalPos);
    return -normalize(cross(a, b));
}

vec4 explode(vec4 position, vec3 normal)
{
    float magnitude = 2.0;
    // vec3 direction = normal * sperateDistance* magnitude;
    vec3 direction = normal * ((sin(sperateDistance) + 1.0) * 0.5) * magnitude;
    return position + vec4(direction, 0.0);
}

void main() 
{    
    mat4 mvp = gs_in[0].projectMatrix * gs_in[0].viewMatrix * gs_in[0].modelMatrix;
    vec3 normal = GetNormal();
    vec4 pos;
    for(int i = 0; i < gl_in.length(); i++)
    {
        pos = vec4(gs_in[i].originalPos, 1.0);
        gl_Position = mvp * explode(pos, normal);
        gs_out.FragPos = gs_in[i].FragPos;
        gs_out.Normal = gs_in[i].Normal;
        gs_out.TexCoords = gs_in[i].TexCoords;
        EmitVertex();
    }
    EndPrimitive();
}