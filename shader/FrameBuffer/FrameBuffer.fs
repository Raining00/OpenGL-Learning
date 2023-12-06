#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenTexture;

const float offset = 1.0 / 300.0;  

void main()
{
    // vec2 offsets[9] = vec2[](
    //     vec2(-offset,  offset), // 左上
    //     vec2( 0.0f,    offset), // 正上
    //     vec2( offset,  offset), // 右上
    //     vec2(-offset,  0.0f),   // 左
    //     vec2( 0.0f,    0.0f),   // 中
    //     vec2( offset,  0.0f),   // 右
    //     vec2(-offset, -offset), // 左下
    //     vec2( 0.0f,   -offset), // 正下
    //     vec2( offset, -offset)  // 右下
    // );

    // // sharp
    // float sharpKernel[9] = float[](
    //     -1, -1, -1,
    //     -1,  9, -1,
    //     -1, -1, -1
    // );

    // // blur
    // float blurKernel[9] = float[](
    //     1.0 / 16, 2.0 / 16, 1.0 / 16,
    //     2.0 / 16, 4.0 / 16, 2.0 / 16,
    //     1.0 / 16, 2.0 / 16, 1.0 / 16
    // );

    // // Edge-detection
    // float edgeKernel[9] = float[](
	// 	-1, -1, -1,
	// 	-1,  8, -1,
	// 	-1, -1, -1
	// );

    // vec3 col = vec3(0.0);
    // for(int i = 0; i < 9; i++)
    // {
    //     col += vec3(texture(screenTexture, TexCoords.st + offsets[i])) * blurKernel[i];
    // }       

    // FragColor = vec4(col, 1.0);

    FragColor = texture(screenTexture, TexCoords);
    float average = 0.2126 * FragColor.r + 0.7152 * FragColor.g + 0.0722 * FragColor.b;
    FragColor = vec4(average, average, average, 1.0);

    vec3 col = texture(screenTexture, TexCoords).rgb;
    FragColor = vec4(col, 1.0);
} 