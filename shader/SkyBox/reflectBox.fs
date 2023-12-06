#version 330 core

struct Material {
    samplerCube diffuse;
    sampler2D specular;
    float shininess;
}; 


out vec4 FragColor;
in vec3 Normal;
in vec3 FragPos;
in vec2 TexCoords;

uniform vec3 cameraPos;
uniform Material material;

void main()
{
    // vec3 norm = normalize(Normal);
    // vec3 viewDir = normalize(cameraPos - FragPos);

    // reflect
    // vec3 I = normalize(FragPos - cameraPos);
    // vec3 R = reflect(I, normalize(Normal));
    // FragColor = vec4(texture(material.diffuse, R).rgb, 1.0);

    // refract
    float ratio = 1.00 / 1.52;
    vec3 I = normalize(FragPos - cameraPos);
    vec3 R = refract(I, normalize(Normal), ratio);
    FragColor = vec4(texture(material.diffuse, R).rgb, 1.0);
}