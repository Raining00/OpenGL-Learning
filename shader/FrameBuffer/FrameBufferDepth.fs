#version 330 core
out vec4 color;
in vec2 TexCoords;

uniform sampler2D screenTexture;

void main()
{             
    float depthValue = texture(screenTexture, TexCoords).r;
    color = vec4(vec3(depthValue), 1.0);
}