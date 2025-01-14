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
static const int WINDOW_HEIGHT = 600;
static const int NUM_PARTICLES = 100000;
static const float DELTA_TIME = 0.01f;
static const int NUM_CELLS = 200;
static const int CELL_SIZE = 2;
static const int MAX_PARTICLES_PER_CELL = 100;

// -----------------------------------------------------------------------------
// Globals
// -----------------------------------------------------------------------------
GLuint g_vbo = 0; // OpenGL VBO
cudaGraphicsResource* g_vboCudaResource = nullptr;

GLuint g_shaderProgram = 0; // Our GLSL shader program
GLint g_scaleLoc = -1; // Uniform location for "uScale"

float2* d_positions = nullptr;
float2* d_velocities = nullptr;

int* d_customFlags = nullptr;
int* d_currentCell = nullptr;
int* d_particlesInCell = nullptr;
int* d_particleCounts = nullptr;

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
__global__ void updatePositionsKernel(float2* pos, int* customFlags, int n, float t)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n)
		return;

	// if(customFlags[i] == 1)
	//     return;
	if(customFlags[i] == 1)
		return;
	pos[i].y -= 9.81f * t;

	// Clamp at ground
	if(pos[i].y < 0.0f)
		pos[i].y = 0.0f;
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

__global__ void calculateCellOccupancy(
	int n, float2* pos, int* currentCells, int* particlesInCell, int* particleCounts)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n)
		return;
	if(pos[i].y <= 0)
	{
		return;
	}
	// Calculate grid cell column and row
	int col = pos[i].x / CELL_SIZE;
	int row = pos[i].y / CELL_SIZE;

	// Calculate the unique cell ID
	int cellId = row * NUM_CELLS + col;

	// Atomically increment particle count for the cell and get the index
	int index = atomicAdd(&particleCounts[cellId], 1);

	// Check if the cell can accommodate more particles
	if(index < MAX_PARTICLES_PER_CELL)
	{
		// Store particle ID in the cell
		particlesInCell[cellId * MAX_PARTICLES_PER_CELL + index] = i;

		// Track the cell ID for the current particle
		currentCells[i] = cellId;

		// Fix the particle's position to the center of the grid cell
		// pos[i].x = col * CELL_SIZE + CELL_SIZE / 2.0f;
		// pos[i].y = row * CELL_SIZE + CELL_SIZE / 2.0f;
	}
}

__global__ void calculateCollisions(int n,
									float2* pos,
									int* currentCells,
									int* particlesInCell,
									int* particleCounts,
									int* customFlags)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= n)
		return;

	int cellId = i;
	// we can now assume this particular thread is dealing with each cell

	// Atomically increment particle count for the cell and get the index
	for(int p1 = 0; p1 < particleCounts[cellId]; p1++)
	{

		int p1_id = particlesInCell[cellId * MAX_PARTICLES_PER_CELL + p1];
		if(p1_id == 0)
			continue;
		for(int p2 = 0; p2 < particleCounts[cellId]; p2++)
		{

			int p2_id = particlesInCell[cellId * MAX_PARTICLES_PER_CELL + p2];

			if(p2_id == 0)
				continue;
			if(p1_id != p2_id)
			{
				float xDiff = abs(pos[p1_id].x - pos[p2_id].x);
				float yDiff = abs(pos[p1_id].y - pos[p2_id].y);

				float distanceSquared = xDiff * xDiff + yDiff * yDiff;
				float collisionDistanceSquared = 4.0f * 0.1 * 0.1; // (2r)^2
				// printf("Cell %d: Particle %d (%.3f, %.3f) vs Particle %d (%.3f, %.3f) - DistSq: %.6f\n",
				//                   cellId, p1_id, pos[p1_id].x, pos[p1_id].y, p2_id, pos[p2_id].x, pos[p2_id].y, distanceSquared);

				if(distanceSquared < collisionDistanceSquared)
				{
					customFlags[p1_id] = 1;
					customFlags[p2_id] = 1;
				}
			}
		}
	}
}
void HandleCollisions()
{

	// first we need to take every particle and put them into
	// to handle this we are going to need more data structures

	{
		dim3 block(256);
		dim3 grid((NUM_PARTICLES + block.x - 1) / block.x);

		// int* d_currentCell = nullptr;
		// int* d_particlesInCell = nullptr;
		// int* d_particleCounts = nullptr;
		calculateCellOccupancy<<<grid, block>>>(
			NUM_PARTICLES, d_positions, d_currentCell, d_particlesInCell, d_particleCounts);
		cudaDeviceSynchronize();
	}

	{
		dim3 block(256);
		dim3 grid((NUM_CELLS * NUM_CELLS + block.x - 1) / block.x);

		// Launch the kernel
		calculateCollisions<<<grid, block>>>(NUM_CELLS * NUM_CELLS, // Total number of cells
											 d_positions,
											 d_currentCell,
											 d_particlesInCell,
											 d_particleCounts,
											 d_customFlags);

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

static GLuint createShaderProgram()
{
	// Minimal vertex shader: just scales position & passes a color
	const char* vsrc = R"(
#version 330 core
        layout(location = 0) in vec2 aPosition;
        uniform vec2 uMin; // Minimum simulation bounds (e.g., [0, 50])
        uniform vec2 uMax; // Maximum simulation bounds (e.g., [100, 100])

        out vec3 vColor;

        void main()
        {
            // Map simulation coordinates to NDC
            vec2 ndcPos = (aPosition - uMin) / (uMax - uMin) * 2.0 - 1.0;
            gl_Position = vec4(ndcPos, 0.0, 1.0);
            gl_PointSize = 2.0f;
            // Simple color mapping (for debugging)
            vColor = vec3((aPosition.x - uMin.x) / (uMax.x - uMin.x), 1.0, (aPosition.y - uMin.y) / (uMax.y - uMin.y));
        }
    )";

	// Minimal fragment shader
	const char* fsrc = R"(
    #version 330 core
    in vec3 vColor;
    out vec4 fragColor;

    void main()
    {
        fragColor = vec4(vColor, 1.0);
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

	// check link
	GLint success;
	glGetProgramiv(prog, GL_LINK_STATUS, &success);
	if(!success)
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
		updatePositionsKernel<<<grid, block>>>(d_positions, d_customFlags, NUM_PARTICLES, g_time);
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
static void displayCallback()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Use our shader
	glUseProgram(g_shaderProgram);
	float simMin[2] = {0.0f, 0.0f}; // Replace with your actual simulation bounds
	float simMax[2] = {100.0f, 100.0f}; // Replace with your actual simulation bounds

	GLint uMinLoc = glGetUniformLocation(g_shaderProgram, "uMin");
	GLint uMaxLoc = glGetUniformLocation(g_shaderProgram, "uMax");

	glUniform2f(uMinLoc, simMin[0], simMin[1]);
	glUniform2f(uMaxLoc, simMax[0], simMax[1]);
	// Set a uniform, e.g., scale
	float scaleValue = 1.0f; // just an example
	glUniform1f(g_scaleLoc, scaleValue);

	// Bind the VBO
	glBindBuffer(GL_ARRAY_BUFFER, g_vbo);

	// Setup attribute 0 for 2D positions
	glEnableVertexAttribArray(0);
	glEnable(GL_PROGRAM_POINT_SIZE);
	glVertexAttribPointer(0, // index = 0
						  2, // vec2
						  GL_FLOAT,
						  GL_FALSE,
						  sizeof(float2),
						  (void*)0);

	// Draw as points
	glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);

	// Cleanup
	glDisableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glutSwapBuffers();
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
	CUDA_CHECK(cudaMalloc((void**)&d_particleCounts, NUM_PARTICLES * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_particlesInCell,
						  NUM_CELLS * NUM_CELLS * MAX_PARTICLES_PER_CELL * sizeof(int)));

	CUDA_CHECK(cudaMalloc((void**)&d_positions, NUM_PARTICLES * sizeof(float2)));
	{
		std::vector<float2> h_positions(NUM_PARTICLES);
		std::vector<int> h_customFlags(NUM_PARTICLES);

		// Simple random range: x in [0..100], y in [50..100]
		srand((unsigned)time(nullptr));
		for(int i = 0; i < NUM_PARTICLES; i++)
		{
			float rx = static_cast<float>(rand() % 101); // 0..100
			float ry = 50.f + static_cast<float>(rand() % 51); // 50..100
			float x = rand() % (NUM_CELLS * CELL_SIZE); // Random x position in the grid
			float y = rand() % (NUM_CELLS * CELL_SIZE); // Random y position in the grid
			h_positions[i] = {x, y};
			int custom = 0;
			if(i > 500 && i < 1000)
				custom = 1;
			h_customFlags[i] = custom;
		}

		// Copy to GPU
		CUDA_CHECK(cudaMemcpy(d_positions,
							  h_positions.data(),
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
