#version 330

out vec4 FragColor;
in vec3 Normal;
in vec3 FragPos;

uniform vec3 toyColor;
uniform vec3 lightColor;
uniform vec3 lightPos;
uniform vec3 viewPos;

// phong lighting model
uniform float ambientStrength;
uniform float diffuseStrength;
uniform float specularStrength;

void main()
{
    // ambient
    vec3 ambient = ambientStrength * lightColor;
  	
    // diffuse 
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diffuseStrength * diff * lightColor;
    
    // specular
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;  
        
    vec3 result = (ambient + diffuse + specular) * toyColor;
    FragColor = vec4(result, 1.0);
}