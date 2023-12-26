#version 330 core

uniform sampler2D image;
uniform vec3 baseColor;

layout(location = 0) out vec4 FragColor;
layout (location = 1) out vec4 BrightColor;

void main()
{
    vec2 imageSize = textureSize(image, 0);
    vec4 color = (0.6 + 0.4 * vec4(1.0, 1.0, 1.0, 1.0)) * texture2D(image, gl_FragCoord.xy / imageSize);
    FragColor = color * vec4(baseColor,1.0f);
    // FragColor = vec4(1.0, 0.2, 0.3, 1.0f);
    float brightness = dot(FragColor.rgb,  vec3(0.2126, 0.7152, 0.0722));
    if(brightness > 1.0)
        BrightColor = vec4(FragColor.rgb, 1.0);
}