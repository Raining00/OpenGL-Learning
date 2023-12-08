#version 330 core

struct Material {
    sampler2D diffuse;
    sampler2D specular;
    sampler2D normal;
    sampler2D height;
    samplerCube reflection;
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
in VS_OUT
{
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
}fs_in;


uniform vec3 cameraPos;

uniform Material material;
uniform DirLight sunLight;

uniform bool UseBlingPhone;
uniform bool GammaCorrectOn;
uniform float gamma;
uniform float halfScreenWidth;

uniform bool compareDifferent;

vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir)
{
    vec3 color = texture(material.diffuse, fs_in.TexCoords).rgb;
    
    vec3 lightDir = normalize(light.direction);
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    // specular shading
    float spec;
    if(!compareDifferent)
    {
        if (UseBlingPhone)
        {
            vec3 halfwayDir = normalize(light.direction + viewDir);
            spec = pow(max(dot(normal, halfwayDir), 0.0), 64.0);
        }
        else
        {
            vec3 reflectDir = reflect(-lightDir, normal);
            spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        }
    }
    else
    {
        if(gl_FragCoord.x < halfScreenWidth)
        {
            vec3 halfwayDir = normalize(light.direction + viewDir);
            spec = pow(max(dot(normal, halfwayDir), 0.0), 64.0);
        }
        else
        {
            vec3 reflectDir = reflect(-lightDir, normal);
            spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        }
    }
    vec3 ambient  = light.ambient  * color;
    vec3 diffuse  = light.diffuse  * diff * color;
    vec3 specular = light.specular * spec * color;
    
    return (ambient + diffuse + specular);
}

void main()
{
    vec3 norm = normalize(fs_in.Normal);
    vec3 viewDir = normalize(cameraPos - fs_in.FragPos);

    // direction light
    vec3 result = CalcDirLight(sunLight, norm, viewDir);

    // gamma correction
    if(!compareDifferent)
    {
        if(GammaCorrectOn)
            result = pow(result, vec3(1.0/gamma));
    }
    else
    {
        if(gl_FragCoord.x < halfScreenWidth)
        {
            result = pow(result, vec3(1.0/gamma));
        }
    }
    FragColor = vec4(result, 1.0);
}