#version 330 core

struct Material {
    sampler2D diffuse;
    sampler2D specular;
    float shininess;
}; 

in vec2 TexCoords;

uniform Material material;

out vec4 FragColor;

void main()
{
    //vec4 texColor = texture(material.diffuse, TexCoords);
    //if(texColor.a < 0.1)
    //    discard;
    FragColor = texture(material.diffuse, TexCoords);
}