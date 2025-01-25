// clang-format off
#include <GL/glew.h>
#include <GL/freeglut.h>
// clang-format on
#include <cmath>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// -----------------------------------------------------------------------------
// Configurable params
// -----------------------------------------------------------------------------
static const int WINDOW_WIDTH = 800;
static const int WINDOW_HEIGHT = 800;
static const int NUM_PARTICLES = 50000;
static const float DELTA_TIME = 0.1f;
static const int NUM_CELLS = 2000;
static const int CELL_SIZE = 10;
__constant__ float SPRING_CONSTANT = 20.0f;
__constant__ float DAMPING_CONSTANT = 10.0f;
static const int MAX_PARTICLES_PER_CELL = 100;

__constant__ float PARTICLE_RADIUS = 5.0f;
// -----------------------------------------------------------------------------
// Globals
// -----------------------------------------------------------------------------
GLuint g_vbo = 0; // OpenGL VBO
cudaGraphicsResource* g_vboCudaResource = nullptr;

GLuint g_shaderProgram = 0; // Our GLSL shader program
GLint g_scaleLoc = -1; // Uniform location for "uScale"

float2* d_positions = nullptr;
float2* d_velocities = nullptr;

float2* d_Newpositions = nullptr;
float2* d_Newvelocities = nullptr;
int* d_customFlags = nullptr;
int* d_currentCell = nullptr;
int* d_particlesInCell = nullptr;
int* d_particleCounts = nullptr;
void drawGrid()
{
	glColor3f(0.3f, 0.3f, 0.3f); // Set grid color (dim gray)
	glBegin(GL_LINES);

	// Draw vertical grid lines
	for(int i = 0; i <= NUM_CELLS; i++)
	{
		float x = i * CELL_SIZE;
		glVertex2f(x, 0.0f); // Bottom of the screen
		glVertex2f(x, NUM_CELLS * CELL_SIZE); // Top of the screen
	}

	// Draw horizontal grid lines
	for(int j = 0; j <= NUM_CELLS; j++)
	{
		float y = j * CELL_SIZE;
		glVertex2f(0.0f, y); // Left side of the screen
		glVertex2f(NUM_CELLS * CELL_SIZE, y); // Right side of the screen
	}

	glEnd();
}
// -----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                                           \
	do                                                                                             \
	{                                                                                              \
		cudaError_t err = call;                                                                    \
		if(err != cudaSuccess)                                                                     \
		{                                                                                          \
			std::cerr << "CUDA Error " << __FILE__ << ":" << __LINE__ << " : "                     \
					  << cudaGetErrorString(err) << std::endl;                                     \
			exit(1);                                                                               \
		}                                                                                          \
	} while(0)

// -----------------------------------------------------------------------------
// 1. CUDA kernel to update positions
// -----------------------------------------------------------------------------

__global__ void
updatePositionsKernel(float2* pos, float2* velocities, int* customFlags, int n, float t)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n)
		return;

	// Skip flagged particles
	if(customFlags[i] == 1)
		return;

	// Apply gravity
	float gravity = -9.81f;

	velocities[i].y += gravity * t;

	// Limit maximum velocity
	float maxVelocity = 20.0f;
	float speedSquared = velocities[i].x * velocities[i].x + velocities[i].y * velocities[i].y;
	if(speedSquared > maxVelocity * maxVelocity)
	{
		float scale = maxVelocity / sqrtf(speedSquared);
		velocities[i].x *= scale;
		velocities[i].y *= scale;
	}
	// Update position using velocity
	pos[i].x += velocities[i].x * t;
	pos[i].y += velocities[i].y * t;

	// Handle collisions with boundaries
	if(pos[i].y <= PARTICLE_RADIUS)
	{
		pos[i].y = PARTICLE_RADIUS;
		velocities[i].y *= -0.8f; // Dampen bounce
		// velocities[i].y = 0.0; // Dampen bounce
	}

	// printf("particle %d with pos { %f, %f} part radius %f  \n",i,pos[i].x,pos[i].y, PARTICLE_RADIUS);
	if(pos[i].y > 20000.0f - PARTICLE_RADIUS)
	{
		pos[i].y = 20000.0f - PARTICLE_RADIUS;
		// velocities[i].y *= -0.4f; // Dampen bounce
	}
	if(pos[i].x <= PARTICLE_RADIUS)
	{
		pos[i].x = PARTICLE_RADIUS;
		velocities[i].x *= -0.8f; // Dampen bounce
		// velocities[i].x *= -0.4f; // Dampen bounce
	}
	if(pos[i].x > 20000.0f - PARTICLE_RADIUS)
	{
		pos[i].x = 20000.0f - PARTICLE_RADIUS;
		velocities[i].x *= -0.8f; // Dampen bounce
	}
}
// -----------------------------------------------------------------------------
// 2. Copy the updated positions from d_positions -> VBO (device->device)
// -----------------------------------------------------------------------------
__global__ void copyToVboKernel(float2* dest, const float2* src, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n)
	{
		dest[i] = src[i];
	}
}
__global__ void resetParticlesCounts(int* particleCounts, int numElements)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < numElements)
	{
		particleCounts[i] = 0; // Use -1 to represent an empty slot
	}
}

__global__ void resetDeltas(float2* positions, float2* velocities, int numElements)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < numElements)
	{
		positions[i] = {0, 0}; // Use -1 to represent an empty slot
		velocities[i] = {0, 0}; // Use -1 to represent an empty slot
	}
}

__global__ void updatePositionsAndVelocities(float2* positions,
											 float2* velocities,
											 float2* new_positions,
											 float2* new_velocities,
											 int numElements)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < numElements)
	{
		positions[i].x = positions[i].x + new_positions[i].x;
		positions[i].y = positions[i].y + new_positions[i].y;
		velocities[i].x = velocities[i].x + new_velocities[i].x;
		velocities[i].y = velocities[i].y + new_velocities[i].y;
	}
}
__global__ void calculateCellOccupancy(int n,
									   float2* pos,
									   int* currentCells,
									   int* particlesInCell,
									   int* particleCounts,
									   int* customFlags)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// printf("i %d \n", i);
	if(i >= n)
		return;
	// printf("pos y %f \n", pos[i].y);
	if(pos[i].y <= 0)
	{
		return;
	}
	// Calculate grid cell column and row

	int col = pos[i].x / CELL_SIZE;
	int row = pos[i].y / CELL_SIZE;
	if(row >= NUM_CELLS || col >= NUM_CELLS || col < 0 || row < 0)
		return;
	// Calculate the unique cell ID
	int cellId = row * NUM_CELLS + col;

	// printf("col %d row %d max for both %d  the cellid is %d \n",col,row, NUM_CELLS, cellId);
	// Atomically increment particle count for the cell and get the index
	// Check if the cell can accommodate more particles

	// if(customFlags[i] == 0)
	//     printf("particle %d with pos { %f, %f} at col %d row %d max for both %d  the cellid is %d \n",i,pos[i].x,pos[i].y,col,row, NUM_CELLS, cellId);

	int index = atomicAdd(&particleCounts[cellId], 1);
	if(index < MAX_PARTICLES_PER_CELL)
	{
		int t = cellId * MAX_PARTICLES_PER_CELL + index;
		// Store particle ID in the cell
		particlesInCell[cellId * MAX_PARTICLES_PER_CELL + index] = i;

		// Track the cell ID for the current particle
		currentCells[i] = cellId;

		// Fix the particle's position to the center of the grid cell
		// pos[i].x = col * CELL_SIZE + CELL_SIZE / 2.0f;
		// pos[i].y = row * CELL_SIZE + CELL_SIZE / 2.0f;
	}
}

__global__ void calculateCollisionsWithDEM(int numParticles,
										   float2* positions,
										   float2* velocities,
										   float2* new_positions,
										   float2* new_velocities,
										   int* currentCells,
										   int* particlesInCell,
										   int* particleCounts)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= numParticles)
		return;

	// Get the current particle's cell ID
	int cellId = currentCells[i];
	int cellX = cellId % NUM_CELLS;
	int cellY = cellId / NUM_CELLS;

	// Current particle properties
	float2 p1_pos = positions[i];
	float2 p1_velocity = velocities[i];

	float particleRadiusSquared = PARTICLE_RADIUS * PARTICLE_RADIUS;
	float collisionDistanceSquared = 4.0f * particleRadiusSquared;

	float2 accumulatedForce = {0.0f, 0.0f};
	float2 accumulatedRepulsion = {0.0f, 0.0f};

	// Loop over all neighboring cells
	for(int offsetY = -1; offsetY <= 1; ++offsetY)
	{
		for(int offsetX = -1; offsetX <= 1; ++offsetX)
		{
			int neighborCellX = cellX + offsetX;
			int neighborCellY = cellY + offsetY;

			// Skip invalid neighbor cells
			if(neighborCellX < 0 || neighborCellX >= NUM_CELLS || neighborCellY < 0 ||
			   neighborCellY >= NUM_CELLS)
				continue;

			int neighborCellId = neighborCellY * NUM_CELLS + neighborCellX;

			// Loop over particles in the neighboring cell
			for(int p2Idx = 0; p2Idx < particleCounts[neighborCellId]; ++p2Idx)
			{
				int p2_id = particlesInCell[neighborCellId * MAX_PARTICLES_PER_CELL + p2Idx];

				// Skip self-interaction
				if(i == p2_id)
					continue;

				float2 p2_pos = positions[p2_id];
				float2 p2_velocity = velocities[p2_id];

				// Calculate the distance squared between particles
				float xDiff = p1_pos.x - p2_pos.x;
				float yDiff = p1_pos.y - p2_pos.y;
				float distanceSquared = xDiff * xDiff + yDiff * yDiff;

				if(distanceSquared < collisionDistanceSquared)
				{
					float distance = sqrtf(distanceSquared);
					float overlap = fmaxf(2.0f * PARTICLE_RADIUS - distance, 0.0f);

					// Collision normal
					float2 collisionNormal = {(p2_pos.x - p1_pos.x) / distance,
											  (p2_pos.y - p1_pos.y) / distance};

					// Relative velocity
					float2 relativeVelocity = {p2_velocity.x - p1_velocity.x,
											   p2_velocity.y - p1_velocity.y};

					// DEM Spring force
					float springForceMagnitude = SPRING_CONSTANT * overlap;
					float2 springForce = {collisionNormal.x * springForceMagnitude,
										  collisionNormal.y * springForceMagnitude};

					// DEM Dashpot force
					float relativeVelocityAlongNormal =
						fminf(relativeVelocity.x * collisionNormal.x +
								  relativeVelocity.y * collisionNormal.y,
							  0.0f);
					float dashpotForceMagnitude = DAMPING_CONSTANT * relativeVelocityAlongNormal;
					float2 dashpotForce = {collisionNormal.x * dashpotForceMagnitude,
										   collisionNormal.y * dashpotForceMagnitude};

					// Accumulate forces
					float sf = 1.0f;
					accumulatedForce.x += sf * (springForce.x + dashpotForce.x);
					accumulatedForce.y += sf * (springForce.y + dashpotForce.y);

					float2 repulsion = {collisionNormal.x * overlap / 2.0f,
										collisionNormal.y * overlap / 2.0f};
					accumulatedRepulsion.x += repulsion.x;
					accumulatedRepulsion.y += repulsion.y;
				}
			}
		}
	}

	// Apply accumulated changes
	new_velocities[i].x += accumulatedForce.x;
	new_velocities[i].y += accumulatedForce.y;
	new_positions[i].x -= accumulatedRepulsion.x;
	new_positions[i].y -= accumulatedRepulsion.y;
}
void HandleCollisions()
{

	{

		dim3 block(256);
		dim3 grid((NUM_CELLS * NUM_CELLS + block.x - 1) / block.x);

		resetParticlesCounts<<<grid, block>>>(d_particleCounts,
											  NUM_CELLS * NUM_CELLS); // Total number of cells
		cudaDeviceSynchronize();
	}

	{

		dim3 block(256);
		dim3 grid((NUM_PARTICLES + block.x - 1) / block.x);
		resetDeltas<<<grid, block>>>(
			d_Newpositions, d_Newvelocities, NUM_PARTICLES); // Total number of cells
		cudaDeviceSynchronize();
	}
	// first we need to take every particle and put them into
	// to handle this we are going to need more data structures

	{
		dim3 block(256);
		dim3 grid((NUM_PARTICLES + block.x - 1) / block.x);

		calculateCellOccupancy<<<grid, block>>>(NUM_PARTICLES,
												d_positions,
												d_currentCell,
												d_particlesInCell,
												d_particleCounts,
												d_customFlags);
		cudaDeviceSynchronize();
	}
	{

		dim3 block(256);
		dim3 grid((NUM_PARTICLES + block.x - 1) / block.x);
		calculateCollisionsWithDEM<<<grid, block>>>(NUM_PARTICLES,
													d_positions,
													d_velocities,

													d_Newpositions,
													d_Newvelocities,
													d_currentCell,
													d_particlesInCell,
													d_particleCounts);

		// Synchronize and free memory
		cudaDeviceSynchronize();
	}

	{
		dim3 block(256);
		dim3 grid((NUM_PARTICLES + block.x - 1) / block.x);
		updatePositionsAndVelocities<<<grid, block>>>(d_positions,
													  d_velocities,
													  d_Newpositions,
													  d_Newvelocities,
													  NUM_PARTICLES); // Total number of cells
		cudaDeviceSynchronize();
	}
}

// -----------------------------------------------------------------------------
// Create and compile a simple GLSL shader
// -----------------------------------------------------------------------------
static GLuint compileShader(const char* src, GLenum type)
{
	GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, &src, nullptr);
	glCompileShader(shader);

	// check compile errors (minimal)
	GLint success;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if(!success)
	{
		char log[512];
		glGetShaderInfoLog(shader, 512, nullptr, log);
		std::cerr << "Shader compile error:\n" << log << std::endl;
		return 0;
	}
	return shader;
}
// static GLuint createShaderProgram()
// {
// 	// Minimal vertex shader: just scales position & passes a color
// 	const char* vsrc = R"(
//
// #version 330 core
//
// layout(location = 0) in vec2 aPosition;
// uniform vec2 uMin;  // Minimum simulation bounds
// uniform vec2 uMax;  // Maximum simulation bounds
// uniform vec2 uScreenSize; // Screen dimensions in pixels
// uniform float uRadius;    // Radius of particles in simulation units
//
// out vec2 vPointCoord; // Pass point coordinates to the fragment shader
//
// void main()
// {
//     // Map simulation coordinates to NDC
//     vec2 ndcPos = (aPosition - uMin) / (uMax - uMin) * 2.0 - 1.0;
//     gl_Position = vec4(ndcPos, 0.0, 1.0);
//
//     // Calculate point size in screen pixels
//     float radiusInNDC = (uRadius / (uMax.x - uMin.x)) * 2.0;
//     gl_PointSize = radiusInNDC * uScreenSize.x;
//
//     // Pass normalized point coordinates (for circular rendering)
//     vPointCoord = aPosition;
// }
//
//     )";
//
// 	// Minimal fragment shader
// 	const char* fsrc = R"(
// #version 330 core
//
// in vec2 vPointCoord; // Point coordinates from vertex shader
//
// uniform float uRadius; // Particle radius
// out vec4 FragColor;    // Output color
//
// void main()
// {
//     // Compute distance from the center of the point
//     float dist = length(gl_PointCoord - vec2(0.5));
//     // if (dist > 0.5) {
//     //     discard; // Discard fragments outside the radius
//     // }
//
//     // Set the particle color (for now, constant green for debugging)
//     FragColor = vec4(0.0, 1.0, 0.0, 1.0);
// }
//     )";
//
// 	// Compile
// 	GLuint vs = compileShader(vsrc, GL_VERTEX_SHADER);
// 	GLuint fs = compileShader(fsrc, GL_FRAGMENT_SHADER);
//
// 	// Link
// 	GLuint prog = glCreateProgram();
// 	glAttachShader(prog, vs);
// 	glAttachShader(prog, fs);
// 	glLinkProgram(prog);
//
// 	// check link
// 	GLint success;
// 	glGetProgramiv(prog, GL_LINK_STATUS, &success);
// 	if(!success)
// 	{
// 		char log[512];
// 		glGetProgramInfoLog(prog, 512, nullptr, log);
// 		std::cerr << "Shader link error:\n" << log << std::endl;
// 		return 0;
// 	}
//
// 	// We can delete the shaders after linking
// 	glDeleteShader(vs);
// 	glDeleteShader(fs);
//
// 	return prog;
// }

static GLuint createShaderProgram()
{
	// Minimal vertex shader: just scales position & passes a color
	const char* vsrc = R"(

#version 330 core

layout(location = 0) in vec2 aPosition;
uniform vec2 uMin;  // Minimum simulation bounds
uniform vec2 uMax;  // Maximum simulation bounds
uniform vec2 uScreenSize; // Screen dimensions in pixels
uniform float uRadius;    // Radius of particles in simulation units

out vec2 vPointCoord; // Pass point coordinates to the fragment shader

void main()
{
    // Map simulation coordinates to NDC
    vec2 ndcPos = (aPosition - uMin) / (uMax - uMin) * 2.0 - 1.0;
    gl_Position = vec4(ndcPos, 0.0, 1.0);

    // Calculate point size in screen pixels
    float radiusInNDC = (uRadius / (uMax.x - uMin.x)) * 2.0;
    gl_PointSize = radiusInNDC * uScreenSize.x;

    // Pass normalized point coordinates (for circular rendering)
    vPointCoord = aPosition;
}

    )";

	// Minimal fragment shader for rendering circular points
	const char* fsrc = R"(
#version 330 core

in vec2 vPointCoord; // Point coordinates from vertex shader
uniform float uRadius; // Particle radius
out vec4 FragColor;    // Output color

void main()
{
    // Compute the normalized coordinates of the current fragment within the point
    vec2 coord = gl_PointCoord - vec2(0.5); // Center the coordinates around (0.0, 0.0)

    // Compute the distance from the center of the point
    float dist = length(coord);

    // Visualize the distance using color
    // Black at the center, white at the edge
    // FragColor = vec4(vec3(dist), 1.0);

    FragColor = vec4(gl_PointCoord, 0.0, 1.0); // Visualize gl_PointCoord

    FragColor = vec4(0.0, 1.0, 0.0, 1.0);

    // Optional: Highlight fragments outside the circle with red
    if (dist > 0.5) {
        // FragColor = vec4(1.0, 0.0, 0.0, 1.0); // Red for discarded fragments
        // Uncomment to test discard behavior
        // discard;
    }
}

    )";

	// Compile
	GLuint vs = compileShader(vsrc, GL_VERTEX_SHADER);
	GLuint fs = compileShader(fsrc, GL_FRAGMENT_SHADER);

	// Link
	GLuint prog = glCreateProgram();
	glAttachShader(prog, vs);
	glAttachShader(prog, fs);
	glLinkProgram(prog);

	// Check link
	GLint success;
	glGetProgramiv(prog, GL_LINK_STATUS, &success);
	if (!success)
	{
		char log[512];
		glGetProgramInfoLog(prog, 512, nullptr, log);
		std::cerr << "Shader link error:\n" << log << std::endl;
		return 0;
	}

	// We can delete the shaders after linking
	glDeleteShader(vs);
	glDeleteShader(fs);

	return prog;
}

// -----------------------------------------------------------------------------
// Create the VBO and register with CUDA
// -----------------------------------------------------------------------------
static void createVBO(size_t numParticles)
{
	glGenBuffers(1, &g_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
	glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(float2), nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Register buffer object with CUDA
	CUDA_CHECK(
		cudaGraphicsGLRegisterBuffer(&g_vboCudaResource, g_vbo, cudaGraphicsRegisterFlagsNone));
}

// -----------------------------------------------------------------------------
// GLUT callbacks
// -----------------------------------------------------------------------------
static float g_time = 0.0f;

// Idle = update + render
static void idleCallback()
{
	g_time += DELTA_TIME;
	{

		HandleCollisions();
	}

	// 1) Update positions in d_positions with a kernel
	{
		dim3 block(256);
		dim3 grid((NUM_PARTICLES + block.x - 1) / block.x);
		updatePositionsKernel<<<grid, block>>>(
			d_positions, d_velocities, d_customFlags, NUM_PARTICLES, DELTA_TIME); //g_timeh
		cudaDeviceSynchronize();
	}

	// 2) Copy from d_positions -> the VBO
	{
		// Map VBO for CUDA
		CUDA_CHECK(cudaGraphicsMapResources(1, &g_vboCudaResource, 0));

		float2* d_vboPtr = nullptr;
		size_t vboSize = 0;
		CUDA_CHECK(
			cudaGraphicsResourceGetMappedPointer((void**)&d_vboPtr, &vboSize, g_vboCudaResource));

		// Launch a kernel to copy
		dim3 block(256);
		dim3 grid((NUM_PARTICLES + block.x - 1) / block.x);
		copyToVboKernel<<<grid, block>>>(d_vboPtr, d_positions, NUM_PARTICLES);
		cudaDeviceSynchronize();

		// Unmap
		CUDA_CHECK(cudaGraphicsUnmapResources(1, &g_vboCudaResource, 0));
	}

	// 3) Trigger a redisplay
	glutPostRedisplay();
}
// static void displayCallback()
// {
// 	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
// 	// Use our shader
// 	glUseProgram(g_shaderProgram);
// 	float simMin[2] = {0.0f, 0.0f}; // Replace with your actual simulation bounds
// 	float simMax[2] = {400.0f, 400.0f}; // Replace with your actual simulation bounds
//
// 	GLint uMinLoc = glGetUniformLocation(g_shaderProgram, "uMin");
// 	GLint uMaxLoc = glGetUniformLocation(g_shaderProgram, "uMax");
//
// 	glUniform2f(uMinLoc, simMin[0], simMin[1]);
// 	glUniform2f(uMaxLoc, simMax[0], simMax[1]);
// 	// Set a uniform, e.g., scale
// 	float scaleValue = 1.0f; // just an example
// 	glUniform1f(g_scaleLoc, scaleValue);
// GLint screenSizeLoc = glGetUniformLocation(g_shaderProgram, "uScreenSize");
// GLint radiusLoc = glGetUniformLocation(g_shaderProgram, "uRadius");
//
// // Set screen size (e.g., 800x600)
// glUniform2f(screenSizeLoc, 800.0f, 600.0f); // Replace with actual screen dimensions
//
// // Set particle radius in simulation units
// glUniform1f(radiusLoc, 10.0f); // Radius of 10 units
// 	// Draw the grid
// 	drawGrid();
// 	// Bind the VBO
// 	glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
//
// 	// Setup attribute 0 for 2D positions
// 	glEnableVertexAttribArray(0);
// 	glEnable(GL_PROGRAM_POINT_SIZE);
//
// glDisable(GL_DEPTH_TEST);
// 	glVertexAttribPointer(0, // index = 0
// 						  2, // vec2
// 						  GL_FLOAT,
// 						  GL_FALSE,
// 						  sizeof(float2),
// 						  (void*)0);
//
// 	// Draw as points
// 	glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);
//
// 	// Cleanup
// 	glDisableVertexAttribArray(0);
// 	glBindBuffer(GL_ARRAY_BUFFER, 0);
//
// 	glutSwapBuffers();
// }

static void displayCallback()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Use our shader
	glUseProgram(g_shaderProgram);

	// Simulation bounds
	float simMin[2] = {0.0f, 0.0f};
	float simMax[2] = {1000.0f, 1000.0f};

	GLint uMinLoc = glGetUniformLocation(g_shaderProgram, "uMin");
	GLint uMaxLoc = glGetUniformLocation(g_shaderProgram, "uMax");
	glUniform2f(uMinLoc, simMin[0], simMin[1]);
	glUniform2f(uMaxLoc, simMax[0], simMax[1]);

	// Scale uniform
	GLint g_scaleLoc = glGetUniformLocation(g_shaderProgram, "uScale");
	float scaleValue = 1.0f;
	glUniform1f(g_scaleLoc, scaleValue);

	// Screen size and particle radius
	GLint screenSizeLoc = glGetUniformLocation(g_shaderProgram, "uScreenSize");
	GLint radiusLoc = glGetUniformLocation(g_shaderProgram, "uRadius");
	glUniform2f(
		screenSizeLoc, WINDOW_WIDTH, WINDOW_HEIGHT); // Replace with actual screen dimensions
	glUniform1f(radiusLoc, 5.0f); // Radius of 10 units

	// Optional: Enable blending for transparency
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Draw the grid
	// drawGrid();

	// Bind the VBO for particle data
	glBindBuffer(GL_ARRAY_BUFFER, g_vbo);

	// Enable and set position attribute
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float2), (void*)0);

	// Enable and set color attribute (if applicable)
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(
		1, 4, GL_FLOAT, GL_FALSE, sizeof(float2) + sizeof(float4), (void*)(sizeof(float2)));

	// Enable point size
	glDisable(GL_DEPTH_TEST);

	// Draw particles as points

	glEnable(GL_PROGRAM_POINT_SIZE);
glEnable(GL_POINT_SMOOTH);
	glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);

	// Cleanup
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Swap buffers
	glutSwapBuffers();

GLenum err;
while ((err = glGetError()) != GL_NO_ERROR) {
    std::cerr << "OpenGL Error: " << err << std::endl;
}
}
// Keyboard callback, so we can exit
static void keyboardCallback(unsigned char key, int x, int y)
{
	if(key == 27)
	{ // ESC
		// Cleanup
		cudaGraphicsUnregisterResource(g_vboCudaResource);
		glDeleteBuffers(1, &g_vbo);
		glDeleteProgram(g_shaderProgram);

		cudaFree(d_positions);
		cudaFree(d_velocities);

		cudaFree(d_Newpositions);
		cudaFree(d_Newvelocities);
		cudaFree(d_currentCell);
		cudaFree(d_particlesInCell);
		cudaFree(d_particleCounts);

		glutDestroyWindow(glutGetWindow());
		exit(0);
	}
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
	// 1) Init GLUT
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutCreateWindow("CUDA + OpenGL Particles");

	// 2) Init GLEW
	GLenum glewStatus = glewInit();
	if(glewStatus != GLEW_OK)
	{
		std::cerr << "GLEW init failed: " << glewGetErrorString(glewStatus) << std::endl;
		return 1;
	}

	// 3) Create VBO + register with CUDA
	createVBO(NUM_PARTICLES);

	// 4) Allocate d_positions on CUDA
	CUDA_CHECK(cudaMalloc((void**)&d_currentCell, NUM_PARTICLES * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_customFlags, NUM_PARTICLES * sizeof(int)));
	// here we need to calculate the amount of cells
	CUDA_CHECK(cudaMalloc((void**)&d_particleCounts, NUM_CELLS * NUM_CELLS * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_particlesInCell,
						  NUM_CELLS * NUM_CELLS * MAX_PARTICLES_PER_CELL * sizeof(int)));

	CUDA_CHECK(cudaMalloc((void**)&d_positions, NUM_PARTICLES * sizeof(float2)));
	CUDA_CHECK(cudaMalloc((void**)&d_velocities, NUM_PARTICLES * sizeof(float2)));

	CUDA_CHECK(cudaMalloc((void**)&d_Newpositions, NUM_PARTICLES * sizeof(float2)));
	CUDA_CHECK(cudaMalloc((void**)&d_Newvelocities, NUM_PARTICLES * sizeof(float2)));
	{
		std::vector<float2> h_positions(NUM_PARTICLES);
		std::vector<float2> h_velocities(NUM_PARTICLES);
		std::vector<int> h_customFlags(NUM_PARTICLES);

		// Simple random range: x in [0..100], y in [50..100]
		srand((unsigned)time(nullptr));
		// for(int i = 0; i < NUM_PARTICLES; i++)
		// {
		// 	float rx = static_cast<float>(rand() % 101); // 0..100
		// 	float ry = 50.f + static_cast<float>(rand() % 51); // 50..100
		// 	float x = rand() % (NUM_CELLS * CELL_SIZE); // Random x position in the grid
		// 	float y = rand() % (NUM_CELLS * CELL_SIZE); // Random y position in the grid
		// 	h_positions[i] = {x, y};
		// 	h_velocities[i] = {0, 0};
		// 	int custom = 0;
		// 	if(i > 500 && i < 1000)
		// 		custom = 1;
		// 	h_customFlags[i] = custom;
		// }
		// for(int i = 0; i < NUM_PARTICLES; i++)
		// {
		// 	// Random position within the grid
		// 	float x = static_cast<float>(rand() % (NUM_CELLS * CELL_SIZE));
		// 	float y = static_cast<float>(rand() % (NUM_CELLS * CELL_SIZE));
		// 	h_positions[i] = {x, y};
		//
		// 	// Small random velocity
		// 	float vx = (rand() % 200 - 100) / 100.0f; // Random x velocity [-1.0, 1.0]
		// 	float vy = (rand() % 200 - 100) / 100.0f; // Random y velocity [-1.0, 1.0]
		// 	h_velocities[i] = {vx, vy};
		//
		// 	// Custom flag logic
		// 	int custom = (i > 500 && i < 1000) ? 1 : 0;
		// 	h_customFlags[i] = custom;
		// }
		//
		for(int i = 0; i < NUM_PARTICLES; i++)
		{
			if(i < NUM_PARTICLES / 2)
			{
				// Left side of the grid
				float x = static_cast<float>(rand() % (NUM_CELLS / 2 * CELL_SIZE));
				float y = static_cast<float>(rand() % (NUM_CELLS * CELL_SIZE));
				// h_positions[i] = {65, 50};

				// Velocity towards the center
				h_positions[i] = {x, y};
				float vx = static_cast<float>(rand() % 100) / 100.0f +
						   0.5f; // Positive x velocity [0.5, 1.5]
				float vy = (rand() % 200 - 100) / 100.0f; // Random y velocity [-1.0, 1.0]
				h_velocities[i] = {vx, vy};
			}
			else
			{
				// Right side of the grid
				float x = static_cast<float>(rand() % (NUM_CELLS / 2 * CELL_SIZE)) +
						  (NUM_CELLS / 2 * CELL_SIZE);
				float y = static_cast<float>(rand() % (NUM_CELLS * CELL_SIZE));
				h_positions[i] = {x, y};

				// h_positions[i] = {50, 50};
				// Velocity towards the center
				float vx = -static_cast<float>(rand() % 100) / 100.0f -
						   0.5f; // Negative x velocity [-1.5, -0.5]
				float vy = (rand() % 200 - 100) / 100.0f; // Random y velocity [-1.0, 1.0]
				h_velocities[i] = {vx, vy};
			}

			// Custom flag logic
			// int custom = (i > 500 && i < 1000) ? 1 : 0;
			// h_customFlags[i] = custom;
		}
		// Add perturbation to avoid perfect stacking
		for(int i = 0; i < NUM_PARTICLES; i++)
		{
			h_positions[i].x +=
				static_cast<float>(rand() % 10 - 5) / 10.0f; // Small x offset [-0.5, 0.5]
			h_positions[i].y +=
				static_cast<float>(rand() % 10 - 5) / 10.0f; // Small y offset [-0.5, 0.5]
		}
		// Copy to GPU
		CUDA_CHECK(cudaMemcpy(d_positions,
							  h_positions.data(),
							  NUM_PARTICLES * sizeof(float2),
							  cudaMemcpyHostToDevice));

		CUDA_CHECK(cudaMemcpy(d_velocities,
							  h_velocities.data(),
							  NUM_PARTICLES * sizeof(float2),
							  cudaMemcpyHostToDevice));

		CUDA_CHECK(cudaMemcpy(d_customFlags,
							  h_customFlags.data(),
							  NUM_PARTICLES * sizeof(int),
							  cudaMemcpyHostToDevice));
	}
	// 5) Create shader program
	g_shaderProgram = createShaderProgram();
	if(!g_shaderProgram)
	{
		std::cerr << "Failed to create shader program.\n";
		return 1;
	}
	g_scaleLoc = glGetUniformLocation(g_shaderProgram, "uScale");

	// 6) Setup some GL state
	glClearColor(0.f, 0.f, 0.f, 1.f);

	// 7) Register GLUT callbacks
	glutDisplayFunc(displayCallback);
	glutIdleFunc(idleCallback);
	glutKeyboardFunc(keyboardCallback);

	// 8) Enter main loop
	glutMainLoop();

	return 0;
}
