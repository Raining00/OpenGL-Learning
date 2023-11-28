#version 330
struct Material {
    sampler2D diffuse;
    sampler2D specular;
    float shininess;
}; 

struct DirLight {
    vec3 direction;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct PointLight {
    vec3 position;

    float constant;
    float linear;
    float quadratic;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct SpotLight {
    vec3 position;
    vec3 direction;
    float innerCutOff;
    float outerCutOff;

    float constant;
    float linear;
    float quadratic;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

out vec4 FragColor;
in vec3 Normal;
in vec3 FragPos;
in vec2 TexCoords;

uniform vec3 cameraPos;

uniform Material material;
uniform DirLight dirLight;
uniform PointLight pointLight;
uniform SpotLight spotLight;
void main()
{
    // Direction light
    // ambient
    // vec3 ambient = dirLight.ambient * vec3(texture(material.diffuse, TexCoords));
    // // diffuse 
    // vec3 norm = normalize(Normal);
    // vec3 lightDir = normalize(-dirLight.direction);
    // float diff = max(dot(norm, lightDir), 0.0);
    // vec3 diffuse = dirLight.diffuse * diff * vec3(texture(material.diffuse, TexCoords));
    // // specular
    // vec3 viewDir = normalize(cameraPos - FragPos);
    // vec3 reflectDir = reflect(-lightDir, norm);  
    // float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    // vec3 specular = spec * dirLight.specular * vec3(texture(material.specular, TexCoords));  
    // vec3 result = ambient + diffuse + specular;

    // // PointLight
    // float distance = length(pointLight.position - FragPos);
    // float attenuation = 1.0 / (pointLight.constant + pointLight.linear * distance + pointLight.quadratic * (distance * distance));
    // ambient = pointLight.ambient * vec3(texture(material.diffuse, TexCoords));
    // lightDir = normalize(pointLight.position - FragPos);
    // diff = max(dot(norm, lightDir), 0.0);
    // diffuse = pointLight.diffuse * diff * vec3(texture(material.diffuse, TexCoords));
    // reflectDir = reflect(-lightDir, norm);
    // spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    // specular = spec * pointLight.specular * vec3(texture(material.specular, TexCoords));
    // result += (ambient + diffuse + specular) * attenuation;

    // SpotLight
    // ambient
    vec3 ambient = spotLight.ambient * texture(material.diffuse, TexCoords).rgb;
    
    // diffuse 
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(spotLight.position - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = spotLight.diffuse * diff * texture(material.diffuse, TexCoords).rgb;  
    
    // specular
    vec3 viewDir = normalize(cameraPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    vec3 specular = spotLight.specular * spec * texture(material.specular, TexCoords).rgb;  
    
    // spotlight (soft edges)
    float theta = dot(lightDir, normalize(-spotLight.direction)); 
    float epsilon = (spotLight.innerCutOff - spotLight.outerCutOff);
    float intensity = clamp((theta - spotLight.outerCutOff) / epsilon, 0.0, 1.0);
    diffuse  *= intensity;
    specular *= intensity;
    
    // attenuation
    float distance    = length(spotLight.position - FragPos);
    float attenuation = 1.0 / (spotLight.constant + spotLight.linear * distance + spotLight.quadratic * (distance * distance));    
    ambient  *= attenuation; 
    diffuse   *= attenuation;
    specular *= attenuation;   
        
    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result, 1.0);
}