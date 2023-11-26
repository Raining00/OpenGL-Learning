#version 330

out vec4 FragColor;
uniform vec3 toyColor;

void main()
{
    FragColor = vec4(toyColor, 1.0f);
}