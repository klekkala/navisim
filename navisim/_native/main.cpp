#include <iostream>
#include <thread>
#include <queue>
#include <map>

#define GLM_FORCE_CUDA
#include "thirdparty/glm/glm/glm.hpp"
#include "thirdparty/glm/glm/vec3.hpp"						// glm::vec3
#include "thirdparty/glm/glm/vec4.hpp"						// glm::vec4
#include "thirdparty/glm/glm/mat4x4.hpp"					// glm::mat4
#include "thirdparty/glm/glm/ext/matrix_transform.hpp"		// glm::translate, glm::rotate, glm::scale
#include "thirdparty/glm/glm/ext/matrix_clip_space.hpp"		// glm::perspective
#include "thirdparty/glm/glm/ext/scalar_constants.hpp"		// glm::pi
#include "thirdparty/glm/glm/gtc/random.hpp"

glm::mat4 camera(float Translate, glm::vec2 const& Rotate)
{
	glm::mat4 Projection = glm::perspective(glm::pi<float>() * 0.25f, 4.0f / 3.0f, 0.1f, 100.f);
	glm::mat4 View = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -Translate));
	View = glm::rotate(View, Rotate.y, glm::vec3(-1.0f, 0.0f, 0.0f));
	View = glm::rotate(View, Rotate.x, glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 Model = glm::scale(glm::mat4(1.0f), glm::vec3(0.5f));
	return Projection * View * Model;
}

void foo() 
{
  // do stuff...
	std::cout << "foo\n" << std::endl;
}

void bar(glm::vec3 Translate, glm::vec3 Rotate, glm::mat4 Projection, glm::mat4* View)
{	
	glm::translate(*View, Translate);

	*View = glm::rotate(*View, Rotate.x, glm::vec3(1.0, 0.0f, 0.0f));
	*View = glm::rotate(*View, Rotate.y, glm::vec3(0.0, 1.0f, 0.0f));
	*View = glm::rotate(*View, Rotate.z, glm::vec3(0.0, 0.0f, 1.0f));
}

int main() 
{
	glm::vec3 Translate = glm::vec3(
		glm::linearRand(0.0f, 1.0f),
		glm::linearRand(0.0f, 1.0f),
		glm::linearRand(0.0f, 1.0f));
	glm::vec3 Rotate = glm::vec3(
		glm::linearRand(0.0f, 1.0f),
		glm::linearRand(0.0f, 1.0f),
		glm::linearRand(0.0f, 1.0f));
	glm::mat4 Projection = glm::perspective(0.87f, 1.6f, 0.1f, 100.f);
	glm::mat4 View = glm::mat4(1.0f);

  std::thread first (foo);     // spawn new thread that calls foo()
  std::thread second(bar, Translate, Rotate, Projection, &View);  // spawn new thread that calls bar(0)

  std::cout << "main, foo and bar now execute concurrently...\n";

  first.join();                // pauses until first finishes
  second.join();               // pauses until second finishes

  std::cout << "foo and bar completed.\n";

  return 0;
}