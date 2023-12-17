#version 330 core
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
    vec3 eyeSpacePos;
    vec4 FragPosLightSpace;
    mat4 projectMatrix;
}fs_in;

uniform vec3 cameraPos;
#define NR_POINT_LIGHTS 4
uniform int pointLightNum;
uniform PointLight pointLights[NR_POINT_LIGHTS];
uniform SpotLight spotLight;
uniform DirLight sunLight;
uniform vec3 baseColor;
uniform sampler2D image;
uniform float pointSize;

vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir)
{
    vec3 color = baseColor;
    
    vec3 lightDir = normalize(light.direction);
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    // specular shading
    float spec;
    vec3 halfwayDir = normalize(light.direction + viewDir);
    vec3 reflectDir = reflect(-lightDir, normal);
    spec = pow(max(dot(normal, halfwayDir), 0.0), 64.0);
    vec3 ambient  = light.ambient  * color;
    vec3 diffuse  = light.diffuse  * diff * color;
    vec3 specular = light.specular * spec * color;
    
    return (ambient + diffuse + specular);
}


vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir)
{
    vec3 lightDir = normalize(light.position - fragPos);
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    // specular shading
    vec3 halfwayDir = normalize(lightDir + viewDir);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 64.0);
    // attenuation
    float distance    = length(light.position - fragPos);
    // float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    float attenuation = 1.0 / (distance);
    vec3 ambient  = light.ambient  * baseColor;
    vec3 diffuse  = light.diffuse  * diff * baseColor;
    vec3 specular = light.specular * spec * baseColor;
    ambient  *= attenuation;
    diffuse  *= attenuation;
    specular *= attenuation;
    return (ambient + diffuse + specular);
}

vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 viewDir)
{
    vec3 lightDir = normalize(light.position - fs_in.FragPos);
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    // specular shading
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 64.0);
    // attenuation
    float distance    = length(light.position - fs_in.FragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    // spotlight intensity
    float theta = dot(lightDir, normalize(-light.direction)); 
    float epsilon = (light.innerCutOff - light.outerCutOff);
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
    vec3 ambient  = light.ambient  * baseColor;
    vec3 diffuse  = light.diffuse  * diff * baseColor;
    vec3 specular = light.specular * spec * baseColor;
    ambient  *= attenuation * intensity;
    diffuse  *= attenuation * intensity;
    specular *= attenuation * intensity;
    return (ambient + diffuse + specular);
}

uniform float far = 100;
uniform float near = 0.1;

float LinearizeDepth(float depth) 
{
    float z = depth * 2.0 - 1.0; // back to NDC 
    return (2.0 * near * far) / (far + near - z * (far - near));    
}

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 brightColor;

void main()
{
    fragColor = vec4(baseColor, 1.0f);
    vec3 normal;
	normal.xy = gl_PointCoord.xy * vec2(2.0, -2.0) + vec2(-1.0,1.0);
	float mag = dot(normal.xy, normal.xy);
	if(mag > 1.0) discard;
    normal.z = sqrt(1.0 - mag);

    vec3 viewDir = normalize(cameraPos - fs_in.FragPos);

    // direction light
    fragColor.xyz = CalcDirLight(sunLight, normal, viewDir);

    for(int i = 0; i < pointLightNum; i++)
    {
        fragColor += vec4(CalcPointLight(pointLights[i], normal, fs_in.FragPos, viewDir), 0.0);
    }

    // gamma correction
    float gamma = 2.2;
    fragColor.rgb = pow(fragColor.rgb, vec3(1.0/gamma));
    float brightness = dot(fragColor.rgb, vec3(0.2126, 0.7152, 0.0722));
    if(brightness > 0.3f)
		brightColor = vec4(fragColor.rgb * brightness, 1.0f);
    vec4 pixelEyePos = vec4(fs_in.eyeSpacePos + vec3(normal.x, normal.y, -normal.z) * pointSize, 1.0f);
	vec4 pixelClipPos = fs_in.projectMatrix * pixelEyePos;
	gl_FragDepth = pixelClipPos.z / pixelClipPos.w;
}