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

// Grid constants
const int WIDTH = 1024;
const int HEIGHT = 768;
const int SIZE = WIDTH * HEIGHT;
int numOfSpecies;

// OpenCL global objects
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue clqueue;
cl_program program;
cl_mem currentGrid;
cl_mem nextGrid;

// OpenGL Pixel Buffer Object and OpenCL buffer for it
GLuint pbo;
cl_mem clPbo;

// Kernel source, This is the code that is compiled by OpenCL to be executed on the GPU
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
)CLC";

// Function prototypes for functions that are necessary
void setupOpenCL();
void cleanupOpenCL();
void runKernel(const string &kernelName, int numOfSpecies);
void initGrid(int numOfSpecies);
void setupPboBuffer(int width, int height);
void displayGrid();
void runSimulation();

// Function prototypes for different Idle functions, Only 1 of them need to be registered as an OpenGL call back function
void idleUncappedFPS();
void idleTimeMillionIterations();
void idleCappedFPS();

// Function protype for functions that aren't necessary
void printGrid();

int main(int argc, char **argv) {
    // Randomly decide number of species
    random_device rd;
    default_random_engine generator(rd());
    uniform_int_distribution<int> speciesGenerator(5, 10);
    numOfSpecies = speciesGenerator(generator);
    cout << "Number of species: " << numOfSpecies << endl;

    // Setup OpenGL
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Game Of Life Using OpenCL and OpenGL");

    // Make sure OpenGL context is current (GLUT should do this automatically)
    // Initialize GLEW (if needed)
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        cerr << "GLEW Error: " << glewGetErrorString(err) << endl;
        return 1;
    }

    setupOpenCL();

    cout << "Initializing Grid ...\n";
    initGrid(numOfSpecies);

    setupPboBuffer(WIDTH, HEIGHT);

    // Register GLUT callback functions
    glutDisplayFunc(displayGrid);
    glutIdleFunc(idleCappedFPS);

    // Start main GLUT loop
    glutMainLoop();

    cleanupOpenCL();
    return 0;
}

void setupOpenCL() {
    cl_int err;

    // Get platform
    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    platform = platforms[0];

    // Get GPU device
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    // Get current OpenGL context and share group
    CGLContextObj glContext = CGLGetCurrentContext();
    CGLShareGroupObj sharegroup = CGLGetShareGroup(glContext);

    // Context properties for OpenCL/OpenGL interop
    cl_context_properties props[] = {
        CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
        (cl_context_properties)sharegroup,
        0
    };

    // Create OpenCL context with share group
    context = clCreateContext(props, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        cerr << "Failed to create OpenCL context: " << err << endl;
        exit(1);
    }


    // Create queue
    clqueue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

    // Create buffer
    currentGrid = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * SIZE, nullptr, &err);
    nextGrid = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * SIZE, nullptr, &err);

    // Compile program
    program = clCreateProgramWithSource(context, 1, &KernelSource, nullptr, &err);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    // Print build log
    size_t logSize;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
    if (logSize > 1) {
        vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        cout << "Build Log:\n" << log.data() << endl;
    }
}

void cleanupOpenCL() {
    clReleaseMemObject(currentGrid);
    clReleaseMemObject(nextGrid);
    clReleaseProgram(program);
    clReleaseCommandQueue(clqueue);
    clReleaseContext(context);
}


void runKernel(const string &kernelName, int numOfSpecies) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, kernelName.c_str(), &err);

    random_device rd;
    default_random_engine generator(rd());
    uniform_int_distribution<int> conflictResolutionGenerator(0, 1);
    int conflictResolution = conflictResolutionGenerator(generator);


    int width = WIDTH;
    int height = HEIGHT;
    err = clEnqueueAcquireGLObjects(clqueue, 1, &clPbo, 0, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &currentGrid);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &nextGrid);
    clSetKernelArg(kernel, 2, sizeof(int), &width);
    clSetKernelArg(kernel, 3, sizeof(int), &height);
    clSetKernelArg(kernel, 4, sizeof(int), &numOfSpecies);
    clSetKernelArg(kernel, 5, sizeof(int), &conflictResolution);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &clPbo);

    size_t globalWorkSize[2] = { (size_t)WIDTH, (size_t)HEIGHT };

    cl_event event;
    clEnqueueNDRangeKernel(clqueue, kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, &event);
    clWaitForEvents(1, &event);

    clEnqueueReleaseGLObjects(clqueue, 1, &clPbo, 0, nullptr, nullptr);
    clFinish(clqueue);

    // Get execution time in nanoseconds
    cl_ulong timeStart, timeEnd;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(timeStart), &timeStart, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(timeEnd), &timeEnd, nullptr);

    double execTimeMS = (timeEnd - timeStart) / 1e6; // convert ns to ms
    double fps = 1000.0 / execTimeMS;


    clReleaseEvent(event);
    clReleaseKernel(kernel);
}

void printGrid() {
    vector<int> grid(SIZE);
    clEnqueueReadBuffer(clqueue, currentGrid, CL_TRUE, 0, sizeof(int) * SIZE, grid.data(), 0, nullptr, nullptr);

    for (int y = 0; y < HEIGHT; y++) { // print only first 5 rows for readability
        for (int x = 0; x < WIDTH; x++) {
            cout << grid[y * WIDTH + x] << "\t";
        }
        cout << endl;
    }
    cout << "...\n";
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
    clEnqueueWriteBuffer(clqueue, currentGrid, CL_TRUE, 0, sizeof(int) * SIZE, grid.data(), 0, nullptr, nullptr);
}

void setupPboBuffer(int width, int height) {
    // Generate PBO
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 3 * sizeof(GLubyte), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Create OpenCL buffer from OpenGL buffer
    cl_int err;
    clPbo = clCreateFromGLBuffer(context, CL_MEM_WRITE_ONLY, pbo, &err);
    if (err != CL_SUCCESS) {
        cerr << "Error creating PBO" << endl;
        exit(1);
    }
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

void runSimulation() {
    // Run kernel to compute the next state that should be displayed
    runKernel("computeNextState", numOfSpecies);

    //Swap currentGrid and NextGrid
    swap(currentGrid, nextGrid);
}

void idleUncappedFPS() {

    // Static variables persist across function calls
    static auto lastTime = chrono::high_resolution_clock::now();
    static int frameCount = 0;

    runSimulation();

    frameCount++;

    // Measure elapsed time
    auto now = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = now - lastTime;
    if (elapsed.count() >= 1.0) {  // every 1 second
        double fps = frameCount / elapsed.count();
        cout << "FPS: " << fps << endl;

        frameCount = 0;
        lastTime = now;
    }


    glutPostRedisplay();
}

void idleTimeMillionIterations() {
    static int frameCount = 0;
    static auto startTime = chrono::high_resolution_clock::now();

    runSimulation();
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

    glutPostRedisplay();
}

void idleCappedFPS() {
    static auto lastTime = chrono::high_resolution_clock::now();
    static int frameCount = 0;

    runSimulation();
    frameCount++;

    const double targetFPS = 35.0;
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