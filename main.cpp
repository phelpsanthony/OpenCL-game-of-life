#include <GL/glew.h>
#include <GL/freeglut.h>
#include <OpenGL/OpenGL.h>
#include <OpenGL/glu.h>
#include <OpenCL/opencl.h>
#include <iostream>
#include <random>
#include <vector>
#include <thread>
#include <chrono>

using namespace std;

const int WIDTH = 1024;
const int HEIGHT = 768;
const int SIZE = WIDTH * HEIGHT;
int numOfSpecies;

cl_platform_id platform_GPU;
cl_device_id device_GPU;
cl_context context;
cl_command_queue clqueue_GPU_compute; // Coma=mand queue for the GPU
cl_command_queue clqueue_GPU_swap;    // Command queue for the CPU kernel that is being simulated on the GPU because I'm a mac user :(
cl_program program_GPU;
cl_mem currentGrid_GPU;
cl_mem nextGrid_GPU;
GLuint pbo;
cl_mem clPbo_GPU;

const char *KernelSource = R"CLC(
__kernel void computeNextState(__global int *currentGrid, __global int *nextGrid, const int width, const int height,
const int numOfSpecies, const int conflictResolution, __global uchar3 *pbo){
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int index = y * width +x;

    if(currentGrid[index] != -1){   // If the cell is alive loop through it's 8 neighbors to count same species neighbors

        int counter = 0;

        for(int i = x-1; i <= x+1; i++)
            for(int j = y-1; j <= y+1; j++){
                // Need to check boundaries of x and y based on 768(y) rows and 1024(x) columns
                if( i<0 || i >= width || j<0 || j>=height) continue;
                if(i==x && j==y) continue;

                if( currentGrid[ j * width + i ] == currentGrid[index])
                    counter++;
            }

        // Set the state of the cell in the nextGrid depending on the number or same species neighbors the cell has
        if (counter == 2 || counter == 3){
            nextGrid[index] = currentGrid[index];
        }
        else
        {
            // -1 is a sentinel value representing a dead cell
            nextGrid[index] = -1;
        }
    }
    else    // if the cell is dead see if the cell has 3 neighbors of any species and come back to life as that species
    {       // if there are 2 species with 3 alive neighbors the conflict must be resolved randomly

        int speciesCount[10] = {0};

        for(int i = x-1; i <= x+1; i++)
            for(int j = y-1; j <= y+1; j++){

                // Check boundaries
                if( i < 0 || i >= width || j < 0 || j >= height) continue;
                if( i == x && j == y) continue;

                int neighborSpeciesId = currentGrid[ j * width + i];
                if( neighborSpeciesId != -1)
                {
                    speciesCount[neighborSpeciesId]++;
                }
            }

        // See if there are any species with 3 alive neighbors to the current cell
        int candidates[2];  // 8 neighbors so a max of 2 species can have 3 alive neighbors to the cell
        int numOfCandidates = 0;

        for(int k = 0; k < numOfSpecies; k++)
        {
            if(speciesCount[k] == 3){
                candidates[numOfCandidates] = k;
                numOfCandidates++;
            }
        }

        // Check for conflict and resolve conflict of necessary
        if(numOfCandidates == 1)
        {
            // Don't need conflict resolution
            nextGrid[index] = candidates[0];
        }
        else if( numOfCandidates == 2)
        {
            // Random conflict resolution required
            nextGrid[index] = candidates[conflictResolution];
        }
        else
        {
            // The cell stays dead
            nextGrid[index] = -1;
        }
    }

    uchar3 color;
    switch(nextGrid[index]){
        case 0: color = (uchar3)(255, 128, 128); break;   // Light Red
        case 1: color = (uchar3)(128, 255, 128); break;   // Light Green
        case 2: color = (uchar3)(128, 128, 255); break;   // Light Blue
        case 3: color = (uchar3)(255, 255, 128); break;   // Light Yellow
        case 4: color = (uchar3)(255, 128, 255); break;   // Light Magenta
        case 5: color = (uchar3)(128, 255, 255); break;   // Light Cyan
        case 6: color = (uchar3)(255, 179, 128); break;   // Light Orange
        case 7: color = (uchar3)(179, 128, 255); break;   // Light Purple
        case 8: color = (uchar3)(204, 204, 204); break;   // Light Gray
        case 9: color = (uchar3)(128, 255, 179); break;   // Light Lime
        default: color = (uchar3)(0, 0, 0); break;  // Black
    }

    pbo[index] = color;

}

// This kernel simulates a CPU kernel which would swap the pointers to the current and next grid
// Because it's a GPU I'm just swapping each cell with each work item and there is 1 work item per cell
// In a CPU I would just swap the pointers like I did in the host program in assignment 3
__kernel void swapGrids(__global int* currentGrid, __global int* nextGrid){
    int idx = get_global_id(0);

    if(idx >= get_global_size(0)) return;

    // Swap the values in each grid
    int temp = currentGrid[idx];
    currentGrid[idx] = nextGrid[idx];
    nextGrid[idx] = temp;
}
)CLC";

void setupOpenCL();
void cleanupOpenCL();
void runKernels();
void initGrid(int numOfSpecies);
void setupPboBuffer(int width, int height);
void displayGrid();
void runSimulation();
void idleCappedFPS();
void idleTimeMillionIterations();
void idleUncappedFPS();

int main(int argc, char **argv){
    random_device rd;
    default_random_engine generator(rd());
    uniform_int_distribution<int> speciesGenerator(5,10);
    numOfSpecies = speciesGenerator(generator);
    cout << "Number of species: " << numOfSpecies << endl;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Game Of Life Using OpenCL and OpenGL");

    GLenum err = glewInit();
    if(GLEW_OK != err){
        cerr << "GLEW Error: " << glewGetErrorString(err) << endl;
        return 1;
    }

    setupOpenCL();
    initGrid(numOfSpecies);
    setupPboBuffer(WIDTH, HEIGHT);

    glutDisplayFunc(displayGrid);
    glutIdleFunc(idleCappedFPS);

    glutMainLoop();
    cleanupOpenCL();
    return 0;
}

void setupOpenCL(){
    cl_int err;
    cl_uint numPlatforms;
    clGetPlatformIDs(0,nullptr,&numPlatforms);
    vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    platform_GPU = platforms[0];

    clGetDeviceIDs(platform_GPU, CL_DEVICE_TYPE_GPU, 1, &device_GPU, nullptr);

    CGLContextObj glContext = CGLGetCurrentContext();
    CGLShareGroupObj sharegroup = CGLGetShareGroup(glContext);

    cl_context_properties props[] = {
        CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
        (cl_context_properties)sharegroup,
        0
    };
    context = clCreateContext(props, 1, &device_GPU, nullptr, nullptr, &err);
    if(err != CL_SUCCESS){ cerr << "Failed to create OpenCL context\n"; exit(1); }

    // Simulate CPU usage by creating a separate command queue for the "CPU kerel"
    clqueue_GPU_compute = clCreateCommandQueue(context, device_GPU, CL_QUEUE_PROFILING_ENABLE, &err);
    clqueue_GPU_swap    = clCreateCommandQueue(context, device_GPU, CL_QUEUE_PROFILING_ENABLE, &err);

    // create buffers to hold the current and next states of the game
    currentGrid_GPU = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*SIZE, nullptr, &err);
    nextGrid_GPU    = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*SIZE, nullptr, &err);

    program_GPU = clCreateProgramWithSource(context,1,&KernelSource,nullptr,&err);
    clBuildProgram(program_GPU,1,&device_GPU,nullptr,nullptr,nullptr);

    size_t logSize;
    clGetProgramBuildInfo(program_GPU,device_GPU,CL_PROGRAM_BUILD_LOG,0,nullptr,&logSize);
    if(logSize>1){
        vector<char> log(logSize);
        clGetProgramBuildInfo(program_GPU,device_GPU,CL_PROGRAM_BUILD_LOG,logSize,log.data(),nullptr);
        cout << "Build Log:\n" << log.data() << endl;
    }
}

void cleanupOpenCL(){
    clReleaseMemObject(currentGrid_GPU);
    clReleaseMemObject(nextGrid_GPU);
    clReleaseProgram(program_GPU);
    clReleaseCommandQueue(clqueue_GPU_compute);
    clReleaseCommandQueue(clqueue_GPU_swap);
    clReleaseContext(context);
}

void runKernels(){
    cl_int err;
    cl_kernel gpuKernel = clCreateKernel(program_GPU,"computeNextState",&err);
    cl_kernel cpuKernel    = clCreateKernel(program_GPU,"swapGrids",&err);

    // generate random number for conflict resolution
    random_device rd;
    default_random_engine generator(rd());
    uniform_int_distribution<int> conflictResolutionGenerator(0,1);
    int conflictResolution = conflictResolutionGenerator(generator);


    // set arguments for the GPU kernel
    clSetKernelArg(gpuKernel,0,sizeof(cl_mem),&currentGrid_GPU);
    clSetKernelArg(gpuKernel,1,sizeof(cl_mem),&nextGrid_GPU);
    clSetKernelArg(gpuKernel,2,sizeof(int),&WIDTH);
    clSetKernelArg(gpuKernel,3,sizeof(int),&HEIGHT);
    clSetKernelArg(gpuKernel,4,sizeof(int),&numOfSpecies);
    clSetKernelArg(gpuKernel,5,sizeof(int),&conflictResolution);
    clSetKernelArg(gpuKernel,6,sizeof(cl_mem),&clPbo_GPU);

    err = clEnqueueAcquireGLObjects(clqueue_GPU_compute,1,&clPbo_GPU,0,nullptr,nullptr);

    size_t global2D[2] = {WIDTH, HEIGHT};
    cl_event computeEvent;
    clEnqueueNDRangeKernel(clqueue_GPU_compute, gpuKernel, 2, nullptr, global2D, nullptr, 0,nullptr,&computeEvent);

    // Swap kernel waits for compute kernel
    clSetKernelArg(cpuKernel,0,sizeof(cl_mem),&currentGrid_GPU);
    clSetKernelArg(cpuKernel,1,sizeof(cl_mem),&nextGrid_GPU);
    size_t global1D = SIZE;
    cl_event swapEvent;

    // Send CPU kernel to command queue and wait for GPU kernel to finish computing next state
    clEnqueueNDRangeKernel(clqueue_GPU_swap, cpuKernel, 1, nullptr, &global1D, nullptr, 1, &computeEvent, &swapEvent);

    clWaitForEvents(1,&swapEvent);

    clEnqueueReleaseGLObjects(clqueue_GPU_compute,1,&clPbo_GPU,0,nullptr,nullptr);
    clFinish(clqueue_GPU_compute);
    clFinish(clqueue_GPU_swap);

    clReleaseEvent(computeEvent);
    clReleaseEvent(swapEvent);
    clReleaseKernel(gpuKernel);
    clReleaseKernel(cpuKernel);
}

void runSimulation(){
    // run the GPU and CPU kernels which compute the next state and then swap the grids
    runKernels();
}

void displayGrid() {
    // Clear the Screen
    glClear(GL_COLOR_BUFFER_BIT);

    // Bind PBO and draw pixels
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Swap buffers
    glutSwapBuffers();
}

void initGrid(int numOfSpecies) {
    random_device rd;
    default_random_engine generator(rd());
    uniform_int_distribution<int> speciesIdGenerator(0, numOfSpecies - 1);

    vector<int> grid(SIZE);
    for (int i = 0; i < SIZE; i++) {
        grid[i] = speciesIdGenerator(generator);
    }

    // Copy the CPU-initialized grid to the GPU buffer
    clEnqueueWriteBuffer(clqueue_GPU_compute, currentGrid_GPU, CL_TRUE, 0, sizeof(int) * SIZE, grid.data(), 0, nullptr, nullptr);
}


void setupPboBuffer(int width, int height) {
    // Generate PBO
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 3 * sizeof(GLubyte), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Create OpenCL buffer from OpenGL buffer
    cl_int err;
    clPbo_GPU = clCreateFromGLBuffer(context, CL_MEM_WRITE_ONLY, pbo, &err);
    if (err != CL_SUCCESS) {
        cerr << "Error creating PBO" << endl;
        exit(1);
    }
}

void idleCappedFPS() {
    static auto lastTime = chrono::high_resolution_clock::now();
    static int frameCount = 0;

    runSimulation();
    frameCount++;

    const double targetFPS = 40.0;
    const double frameTime = 1.0 / targetFPS; // seconds per frame
    auto now = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = now - lastTime;

    if (elapsed.count() < frameTime) {
        // Sleep for remaining time to cap FPS
        auto sleepTime = chrono::duration<double>(frameTime - elapsed.count());
        auto sleepDuration = chrono::duration_cast<chrono::microseconds>(sleepTime);
        this_thread::sleep_for(sleepDuration);
        now = chrono::high_resolution_clock::now(); // update time after sleep
        elapsed = now - lastTime;
    }

    // Print FPS every 1 second
    static auto fpsTimer = chrono::high_resolution_clock::now();
    static int fpsCounter = 0;
    fpsCounter++;

    auto fpsNow = chrono::high_resolution_clock::now();
    chrono::duration<double> fpsElapsed = fpsNow - fpsTimer;
    if (fpsElapsed.count() >= 1.0) {
        double actualFPS = fpsCounter / fpsElapsed.count();
        cout << "Actual FPS: " << actualFPS << endl;

        fpsCounter = 0;
        fpsTimer = fpsNow;
    }

    lastTime = now;
    glutPostRedisplay();
}

void idleTimeMillionIterations() {
    static int frameCount = 0;
    static auto startTime = chrono::high_resolution_clock::now();

    runSimulation();
    glutPostRedisplay();
    frameCount++;

    if (frameCount >= 1000000) {
        auto endTime = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = endTime - startTime;
        double avgFPS = frameCount / elapsed.count();
        cout << "Completed " << frameCount << " frames in "
                     << elapsed.count() << " seconds. Average FPS: "
                     << avgFPS << endl;
        // Reset counters for the next measurement
        frameCount = 0;
        startTime = chrono::high_resolution_clock::now();
    }
    else if (frameCount % 1000 == 0) {
        auto now = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = now - startTime;
        double currentFPS = frameCount / elapsed.count();
        cout << "Progress: " << frameCount << " / 1000000 frames"
             << " | Current FPS: " << currentFPS << endl;
    }


    // glutPostRedisplay();
}

void idleUncappedFPS() {

    // Static variables persist across function calls
    static auto lastTime = chrono::high_resolution_clock::now();
    static int frameCount = 0;

    runSimulation();

    frameCount++;
    glutPostRedisplay();
    // Measure elapsed time
    auto now = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = now - lastTime;
    if (elapsed.count() >= 1.0) {  // every 1 second
        double fps = frameCount / elapsed.count();
        cout << "FPS: " << fps << endl;

        frameCount = 0;
        lastTime = now;
    }


    // glutPostRedisplay();
}