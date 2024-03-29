#version 330 core

uniform sampler2D image;
uniform vec3 baseColor;

layout(location = 0) out vec4 FragColor;
layout (location = 1) out vec4 BrightColor;

in VS_OUT
{
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
    vec3 eyeSpacePos;
    vec4 FragPosLightSpace;
    vec4 Color;
    mat4 projectMatrix;
}fs_in;

void main()
{
    vec3 normal;
	normal.xy = gl_PointCoord.xy * vec2(2.0, -2.0) + vec2(-1.0,1.0);
	float mag = dot(normal.xy, normal.xy);
	if(mag > 1.0) discard;

    vec2 imageSize = textureSize(image, 0);
    vec4 color = (0.6 + 0.4 * fs_in.Color) * texture2D(image, gl_FragCoord.xy);
    FragColor = color * vec4(baseColor,1.0f);
    // FragColor = vec4(1.0, 0.2, 0.3, 1.0f);
    float brightness = dot(FragColor.rgb,  vec3(0.2126, 0.7152, 0.0722));
    // float brightness = dot(FragColor.rgb,  vec3(0.2627, 0.6780, 0.0593));
    if(brightness > 1.0)
        BrightColor = vec4(FragColor.rgb, 1.0);
}