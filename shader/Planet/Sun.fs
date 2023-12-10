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
uniform float time;

out vec4 FragColor;
in VS_OUT
{
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
}fs_in;

float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

void main() {
    // 采样纹理
    vec3 result = texture(material.diffuse, vec2(fs_in.TexCoords.x + time / 1000.f, fs_in.TexCoords.y)).rgb;

    // Gamma校正
    float gamma = 2.2;
    result = pow(result, vec3(1.0/gamma));

    FragColor = vec4(result, 1.0);
}