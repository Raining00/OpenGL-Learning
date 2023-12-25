#version 330 core

layout(location = 0) out vec4 FragColor;
layout (location = 1) out vec4 BrightColor;

struct Material {
    sampler2D diffuse; // 0
    sampler2D specular;// 1
    sampler2D normal;  // 2
    sampler2D height;  // 3
    samplerCube reflection; // 4
    float shininess;
    bool useNormalMap;
}; 

uniform sampler2D shadowMap; // 5
uniform samplerCube shadowDepthCube; // 6

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

in VS_OUT
{
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
    vec4 FragPosLightSpace;
    mat3 TBN;
}fs_in;


uniform vec3 cameraPos;

uniform Material material;

#define NR_POINT_LIGHTS 4
uniform int pointLightNum;
uniform PointLight pointLights[NR_POINT_LIGHTS];
uniform SpotLight spotLight;
uniform DirLight sunLight;

uniform bool receiveShadow;

float near = 0.1; 
uniform float far_plane;

float ShadowCalculation(vec4 FragPosLightSpace)
{
    vec3 projCoords = FragPosLightSpace.xyz / FragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;

    float bias = max(0.05 * (1.0 - dot(fs_in.Normal, sunLight.direction)), 0.005);
    // float shadow = currentDepth > closestDepth ? 1.0 : 0.0;
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; ++x)
    {
        for(int y = -1; y <= 1; ++y)
        {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
            shadow += currentDepth > pcfDepth ? 1.0 : 0.0;        
        }    
    }
    shadow /= 9.0;
    if(projCoords.z > 1.0)
        shadow = 0.0;
    return shadow;
}

float PointShadowCalculation(vec3 fragPos, vec3 lightPos)
{
    vec3 sampleOffsetDirections[20] = vec3[]
    (
       vec3( 1,  1,  1), vec3( 1, -1,  1), vec3(-1, -1,  1), vec3(-1,  1,  1), 
       vec3( 1,  1, -1), vec3( 1, -1, -1), vec3(-1, -1, -1), vec3(-1,  1, -1),
       vec3( 1,  1,  0), vec3( 1, -1,  0), vec3(-1, -1,  0), vec3(-1,  1,  0),
       vec3( 1,  0,  1), vec3(-1,  0,  1), vec3( 1,  0, -1), vec3(-1,  0, -1),
       vec3( 0,  1,  1), vec3( 0, -1,  1), vec3( 0, -1, -1), vec3( 0,  1, -1)
    );
    // get vector between fragment position and light position
    vec3 fragToLight = fragPos - lightPos;
    // ise the fragment to light vector to sample from the depth map    
    float closestDepth = texture(shadowDepthCube, fragToLight).r;
    // it is currently in linear range between [0,1], let's re-transform it back to original depth value
    closestDepth *= far_plane;
    // now get current linear depth as the length between the fragment and light position
    float currentDepth = length(fragToLight);
    // test for shadows
    float shadow = 0.0;
    float bias = 0.15;
    int samples = 20;
    float viewDistance = length(cameraPos - fragPos);
    float diskRadius = (1.0 + (viewDistance / far_plane)) / 25.0;
    for(int i = 0; i < samples; ++i)
    {
        float closestDepth = texture(shadowDepthCube, fragToLight + sampleOffsetDirections[i] * diskRadius).r;
        closestDepth *= far_plane;   // Undo mapping [0;1]
        if(currentDepth - bias > closestDepth)
            shadow += 1.0;
    }
    shadow /= float(samples);
    // if no PCF, use simple comparison to sample from the depth map
    // float shadow = currentDepth  > closestDepth ? 1.0 : 0.0;        
    // display closestDepth as debug (to visualize depth cubemap)
    // FragColor = vec4(vec3(closestDepth / far_plane), 1.0);    
        
    return shadow;
}

vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir)
{
    vec3 color = texture(material.diffuse, fs_in.TexCoords).rgb;
    
    vec3 lightDir = normalize(light.direction);
    // diffuse shading
    if(material.useNormalMap)
    {
        lightDir = fs_in.TBN * lightDir;
		normal = texture(material.normal, fs_in.TexCoords).rgb;
        normal = normalize(normal * 2.0 - 1.0);   
    }
    float diff = max(dot(lightDir, normal), 0.0);
    // specular shading
    vec3 halfwayDir = normalize(light.direction + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 64.0);
    vec3 ambient  = light.ambient  * color;
    vec3 diffuse  = light.diffuse  * diff * color;
    vec3 specular = light.specular * spec * color;
    float shadow = 0.0;
    if(receiveShadow)
        shadow = ShadowCalculation(fs_in.FragPosLightSpace);
    return (ambient + (1.0 - shadow) * (diffuse + specular));
}
bool firstTime = true;

vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir)
{
    vec3 lightDir = normalize(light.position - fragPos);
    if(material.useNormalMap)
    {
        lightDir = fs_in.TBN * lightDir;
        normal = texture(material.normal, fs_in.TexCoords).rgb;
        normal = normalize(normal * 2.0 - 1.0);   
    }
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    // specular shading
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 64.0);
    // attenuation
    float distance    = length(light.position - fragPos);
    // float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    float attenuation = 1.0 / (distance);
    vec3 ambient  = light.ambient  * vec3(texture(material.diffuse, fs_in.TexCoords));
    vec3 diffuse  = light.diffuse  * diff * vec3(texture(material.diffuse, fs_in.TexCoords));
    vec3 specular = light.specular * spec * vec3(texture(material.specular, fs_in.TexCoords));
    ambient  *= attenuation;
    diffuse  *= attenuation;
    specular *= attenuation;
    float shadow = 0.0f;
    if(receiveShadow && pointLightNum > 0)
    {
        if(firstTime)
        {
            shadow = PointShadowCalculation(fragPos, light.position);
            firstTime = false;
        }
    }
    return (ambient + (1-shadow) * (diffuse + specular));
}

vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 viewDir)
{
    vec3 lightDir = normalize(light.position - fs_in.FragPos);
    if(material.useNormalMap)
    {
		normal = texture(material.normal, fs_in.TexCoords).rgb;
		normal = normalize(normal * 2.0 - 1.0);
    }
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    // specular shading
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    // attenuation
    float distance    = length(light.position - fs_in.FragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    // spotlight intensity
    float theta = dot(lightDir, normalize(-light.direction)); 
    float epsilon = (light.innerCutOff - light.outerCutOff);
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
    vec3 ambient  = light.ambient  * vec3(texture(material.diffuse, fs_in.TexCoords));
    vec3 diffuse  = light.diffuse  * diff * vec3(texture(material.diffuse, fs_in.TexCoords));
    vec3 specular = light.specular * spec * vec3(texture(material.specular, fs_in.TexCoords));
    ambient  *= attenuation * intensity;
    diffuse  *= attenuation * intensity;
    specular *= attenuation * intensity;
    return (ambient + diffuse + specular);
}

float LinearizeDepth(float depth) 
{
    float z = depth * 2.0 - 1.0; // back to NDC 
    return (2.0 * near * far_plane) / (far_plane + near - z * (far_plane - near));    
}

void main()
{
    vec3 norm = normalize(fs_in.Normal);
    vec3 viewDir;
    if(material.useNormalMap)
        viewDir = normalize(fs_in.TBN * (cameraPos - fs_in.FragPos));
    else
        viewDir = normalize(cameraPos - fs_in.FragPos);

    // direction light
    vec3 result = CalcDirLight(sunLight, norm, viewDir);

    for(int i = 0; i < pointLightNum; i++)
    {
        result += CalcPointLight(pointLights[i], norm, fs_in.FragPos, viewDir);
    }

    // gamma correction
    float gamma = 2.2;
    result = pow(result, vec3(1.0/gamma));
    FragColor = vec4(result, 1.0);
    float brightness = dot(FragColor.rgb,  vec3(0.2126, 0.7152, 0.0722));
    if(brightness > 1.0)
        BrightColor = vec4(FragColor.rgb, 1.0);
    // depth dest
    // float depth = LinearizeDepth(gl_FragCoord.z) / far;
    // FragColor = vec4(vec3(depth), 1.0);
}