#version 330 core
out vec4 FragColor;

in vec3 fColor;

void main()
{
    // float dist = length(gl_PointCoord - vec2(0.5, 0.5));
    // if(dist > 0.5)
    //     discard;
    FragColor = vec4(fColor, 1.0); 
}