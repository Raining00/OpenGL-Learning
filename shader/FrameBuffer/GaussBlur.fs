#version 330 core

layout(location = 0) out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D image;

uniform bool horizontal;
// bilinear gaussian
uniform float offset[3] = float[](0.0, 1.3846153846, 3.2307692308);
uniform float weight[3] = float[](0.2270270270, 0.3162162162, 0.0702702703);

void main()
{
    vec2 texSize = textureSize(image, 0); 
    vec2 tex_offset = 1.0 / textureSize(image, 0); // gets size of single texel
    vec3 result = texture(image, TexCoords).rgb * weight[0]; // current fragment's contribution
    if(horizontal)
    {
        for(int i = 1; i < 3; ++i)
        {
            result += texture(image, TexCoords + vec2(offset[i] / texSize.x, 0.0)).rgb * weight[i];
            result += texture(image, TexCoords - vec2(offset[i] / texSize.x, 0.0)).rgb * weight[i];
        }
    }
    else
    {
        for(int i = 1; i < 3; ++i)
        {
            result += texture(image, TexCoords + vec2(0.0, offset[i] / texSize.y)).rgb * weight[i];
            result += texture(image, TexCoords - vec2(0.0, offset[i] / texSize.y)).rgb * weight[i];
        }
    }
    FragColor = vec4(result, 1.0);
} 