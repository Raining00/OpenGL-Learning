#version 330 core

struct Material {
    sampler2D diffuse;
    sampler2D specular;
    sampler2D normal;
    sampler2D height;
    samplerCube reflection;
    float shininess;
}; 

uniform Material material;

out vec4 FragColor;
in VS_OUT
{
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
}fs_in;

void main()
{
    // direction light
    vec3 result = texture(material.diffuse, fs_in.TexCoords).rgb;

    //gamma correction
    float gamma = 2.2;
    result = pow(result, vec3(1.0/gamma));
    FragColor = vec4(result, 1.0);
}