#pragma once

#include "Drawable/Drawable.h"
#include "Manager/ShaderManager.h"
#include "Manager//TextureManager.h"

#include "Engine/cuda/core/Array.h"
#include "Engine/cuda/core/Vector/Vector.h"

namespace Renderer
{
	class LightDrawable : public Drawable
	{
	public:
		LightDrawable(const std::string& name);
		LightDrawable(const std::string& name, const glm::vec3& ambient, const glm::vec3& diffuse, const glm::vec3& specular, const float& constant, const float& linear, const float& quadratic, const glm::vec3& position = glm::vec3(0.0f));
		~LightDrawable() = default;

		void setLightColor(const glm::vec3& ambient, const glm::vec3& diffuse, const glm::vec3& specular)
		{
			m_amibent = ambient;
			m_diffuse = diffuse;
			m_specular = specular;
			updateLightSetting();
		}

		void setLightAttenuation(const float& constant, const float& linear, const float& quadratic)
		{
			m_constant = constant;
			m_linear = linear;
			m_quadratic = quadratic;
			updateLightSetting();
		}

		void setLightPosition(const glm::vec3& position)
		{
			getTransformation()->setTranslation(position);
			updateLightSetting();
		}

		void setLightColor(const glm::vec3& color)
		{
			m_lightColor = color;
		}

		virtual void render(Camera3D::ptr camera, Camera3D::ptr lightCamera, Shader::ptr shader = nullptr) override;
		virtual void renderDepth(Shader::ptr shader, Camera3D::ptr lightCamera) override {}
		virtual void renderDepthCube(Shader::ptr shader) override {}

	private:
		virtual void renderImp() override;

	private:
		void updateLightSetting();

	private:
		glm::vec3 m_amibent{ glm::vec3(0.2) };
		glm::vec3 m_diffuse{ glm::vec3(0.5) };
		glm::vec3 m_specular{ glm::vec3(1.0) };
		glm::vec3 m_lightColor{ glm::vec3(1.0) };
		float m_constant{ 1.0f };
		float m_linear{ 0.09f };
		float m_quadratic{ 0.032f };
		std::string m_lightName;
		unsigned int m_shaderIndex;
		unsigned int m_lightIndex;
	};


}