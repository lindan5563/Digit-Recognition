#define _CRT_SECURE_NO_WARNINGS
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <thread>
#include <random>
using namespace std;

// ========== 核心参数 ==========
const int CANVAS_SIZE = 28;
const int PIXEL_COUNT = CANVAS_SIZE * CANVAS_SIZE;
const float PIXEL_DISPLAY_SIZE = 20.0f;

struct Pixel { int gray; };
vector<Pixel> canvas(PIXEL_COUNT);
bool mousePressed[2] = { false, false };
double mouseX = 0, mouseY = 0;
int brushSize = 1;
int eraserSize = 3;
bool showBrushCursor = true;  // 是否显示画笔光标

// 当前画笔位置
int currentBrushX = -1;
int currentBrushY = -1;
bool isInCanvas = false;

unsigned int shaderProgram, VBO, VAO;
unsigned int gridShaderProgram, gridVBO, gridVAO;
unsigned int cursorShaderProgram, cursorVBO, cursorVAO;  // 光标着色器
int screenWidth = 1000, screenHeight = 900;
GLFWwindow* window;
string savefilename;

// ========== 着色器代码 ==========
const char* vertexShaderSource = R"(
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in float aGray;
out float grayValue;
uniform vec2 screenSize;
void main()
{
    float flippedY = screenSize.y - aPos.y;
    vec2 ndc = vec2((aPos.x / screenSize.x) * 2.0 - 1.0, (flippedY / screenSize.y) * 2.0 - 1.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    grayValue = aGray;
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
in float grayValue;
out vec4 FragColor;
void main()
{
    float brightness = grayValue / 255.0;
    FragColor = vec4(brightness, brightness, brightness, 1.0);
}
)";

const char* gridVertexShaderSource = R"(
#version 330 core
layout(location=0) in vec2 aPos;
uniform vec2 screenSize;
void main()
{
    float flippedY = screenSize.y - aPos.y;
    vec2 ndc = vec2((aPos.x / screenSize.x) * 2.0 - 1.0, (flippedY / screenSize.y) * 2.0 - 1.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
}
)";

const char* gridFragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
void main()
{
    FragColor = vec4(0.4, 0.4, 0.4, 0.6);
}
)";

// ========== 光标着色器代码 ==========
const char* cursorVertexShaderSource = R"(
#version 330 core
layout(location=0) in vec2 aPos;
uniform vec2 screenSize;
void main()
{
    float flippedY = screenSize.y - aPos.y;
    vec2 ndc = vec2((aPos.x / screenSize.x) * 2.0 - 1.0, (flippedY / screenSize.y) * 2.0 - 1.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
}
)";

const char* cursorFragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
uniform bool isEraser;
void main()
{
    if (isEraser) {
        FragColor = vec4(0.2, 0.6, 1.0, 0.3);  // 蓝色半透明 - 橡皮
    } else {
        FragColor = vec4(1.0, 0.8, 0.2, 0.3);  // 金色半透明 - 画笔
    }
}
)";

// ========== 初始化函数 ==========
void initializeRenderer()
{
    for (int i = 0; i < PIXEL_COUNT; i++) {
        canvas[i].gray = 0;
    }

    // 初始化主着色器
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // 初始化网格着色器
    unsigned int gridVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(gridVertexShader, 1, &gridVertexShaderSource, NULL);
    glCompileShader(gridVertexShader);

    unsigned int gridFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(gridFragmentShader, 1, &gridFragmentShaderSource, NULL);
    glCompileShader(gridFragmentShader);

    gridShaderProgram = glCreateProgram();
    glAttachShader(gridShaderProgram, gridVertexShader);
    glAttachShader(gridShaderProgram, gridFragmentShader);
    glLinkProgram(gridShaderProgram);

    glDeleteShader(gridVertexShader);
    glDeleteShader(gridFragmentShader);

    glGenVertexArrays(1, &gridVAO);
    glGenBuffers(1, &gridVBO);

    glBindVertexArray(gridVAO);
    glBindBuffer(GL_ARRAY_BUFFER, gridVBO);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // 初始化光标着色器
    unsigned int cursorVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(cursorVertexShader, 1, &cursorVertexShaderSource, NULL);
    glCompileShader(cursorVertexShader);

    unsigned int cursorFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(cursorFragmentShader, 1, &cursorFragmentShaderSource, NULL);
    glCompileShader(cursorFragmentShader);

    cursorShaderProgram = glCreateProgram();
    glAttachShader(cursorShaderProgram, cursorVertexShader);
    glAttachShader(cursorShaderProgram, cursorFragmentShader);
    glLinkProgram(cursorShaderProgram);

    glDeleteShader(cursorVertexShader);
    glDeleteShader(cursorFragmentShader);

    glGenVertexArrays(1, &cursorVAO);
    glGenBuffers(1, &cursorVBO);

    glBindVertexArray(cursorVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cursorVBO);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

// ========== 绘制网格 ==========
void drawGrid()
{
    float totalCanvasWidth = CANVAS_SIZE * PIXEL_DISPLAY_SIZE;
    float totalCanvasHeight = CANVAS_SIZE * PIXEL_DISPLAY_SIZE;
    float startX = (screenWidth - totalCanvasWidth) / 2.0f;
    float startY = (screenHeight - totalCanvasHeight) / 2.0f;
    vector<float> gridVertices;

    // 垂直线
    for (int i = 0; i <= CANVAS_SIZE; i++) {
        float x = startX + i * PIXEL_DISPLAY_SIZE;
        gridVertices.push_back(x);
        gridVertices.push_back(startY);
        gridVertices.push_back(x);
        gridVertices.push_back(startY + totalCanvasHeight);
    }
    // 水平线
    for (int i = 0; i <= CANVAS_SIZE; i++) {
        float y = startY + i * PIXEL_DISPLAY_SIZE;
        gridVertices.push_back(startX);
        gridVertices.push_back(y);
        gridVertices.push_back(startX + totalCanvasWidth);
        gridVertices.push_back(y);
    }

    glUseProgram(gridShaderProgram);
    glUniform2f(glGetUniformLocation(gridShaderProgram, "screenSize"),
        (float)screenWidth, (float)screenHeight);

    glBindVertexArray(gridVAO);
    glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
    glBufferData(GL_ARRAY_BUFFER, gridVertices.size() * sizeof(float),
        gridVertices.data(), GL_STATIC_DRAW);

    glDrawArrays(GL_LINES, 0, gridVertices.size() / 2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

// ========== 绘制光标 ==========
void drawCursor()
{
    if (!showBrushCursor || !isInCanvas || currentBrushX < 0 || currentBrushY < 0) {
        return;
    }

    float totalCanvasWidth = CANVAS_SIZE * PIXEL_DISPLAY_SIZE;
    float totalCanvasHeight = CANVAS_SIZE * PIXEL_DISPLAY_SIZE;
    float startX = (screenWidth - totalCanvasWidth) / 2.0f;
    float startY = (screenHeight - totalCanvasHeight) / 2.0f;

    int radius = mousePressed[1] ? eraserSize : brushSize;
    bool isEraser = mousePressed[1];

    // 计算光标位置（屏幕坐标）
    float centerX = startX + currentBrushX * PIXEL_DISPLAY_SIZE + PIXEL_DISPLAY_SIZE / 2.0f;
    float centerY = startY + currentBrushY * PIXEL_DISPLAY_SIZE + PIXEL_DISPLAY_SIZE / 2.0f;

    // 计算光标半径（屏幕像素）
    float cursorRadius = radius * PIXEL_DISPLAY_SIZE;

    // 生成圆圈的顶点数据（使用线段近似圆）
    vector<float> circleVertices;
    const int segments = 32;

    for (int i = 0; i < segments; i++) {
        float angle1 = 2.0f * 3.14159f * i / segments;
        float angle2 = 2.0f * 3.14159f * (i + 1) / segments;

        float x1 = centerX + cursorRadius * cos(angle1);
        float y1 = centerY + cursorRadius * sin(angle1);
        float x2 = centerX + cursorRadius * cos(angle2);
        float y2 = centerY + cursorRadius * sin(angle2);

        circleVertices.push_back(x1);
        circleVertices.push_back(y1);
        circleVertices.push_back(x2);
        circleVertices.push_back(y2);
    }

    // 绘制内部半透明圆
    vector<float> fillVertices;
    for (int i = 0; i < segments; i++) {
        float angle1 = 2.0f * 3.14159f * i / segments;
        float angle2 = 2.0f * 3.14159f * (i + 1) / segments;

        float x1 = centerX + cursorRadius * cos(angle1);
        float y1 = centerY + cursorRadius * sin(angle1);
        float x2 = centerX + cursorRadius * cos(angle2);
        float y2 = centerY + cursorRadius * sin(angle2);

        fillVertices.push_back(centerX);
        fillVertices.push_back(centerY);
        fillVertices.push_back(x1);
        fillVertices.push_back(y1);
        fillVertices.push_back(x2);
        fillVertices.push_back(y2);
    }

    glUseProgram(cursorShaderProgram);
    glUniform2f(glGetUniformLocation(cursorShaderProgram, "screenSize"),
        (float)screenWidth, (float)screenHeight);
    glUniform1i(glGetUniformLocation(cursorShaderProgram, "isEraser"), isEraser ? 1 : 0);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // 绘制填充
    glBindVertexArray(cursorVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cursorVBO);
    glBufferData(GL_ARRAY_BUFFER, fillVertices.size() * sizeof(float),
        fillVertices.data(), GL_STATIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, fillVertices.size() / 2);

    // 绘制轮廓（更不透明）
    glUniform1i(glGetUniformLocation(cursorShaderProgram, "isEraser"), isEraser ? 1 : 0);
    glBufferData(GL_ARRAY_BUFFER, circleVertices.size() * sizeof(float),
        circleVertices.data(), GL_STATIC_DRAW);
    glDrawArrays(GL_LINES, 0, circleVertices.size() / 2);

    glDisable(GL_BLEND);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

// ========== 绘制像素 ==========
void drawPixels()
{
    float totalCanvasWidth = CANVAS_SIZE * PIXEL_DISPLAY_SIZE;
    float totalCanvasHeight = CANVAS_SIZE * PIXEL_DISPLAY_SIZE;
    float startX = (screenWidth - totalCanvasWidth) / 2.0f;
    float startY = (screenHeight - totalCanvasHeight) / 2.0f;

    // 绘制网格
    drawGrid();

    // 创建像素顶点数据
    vector<float> vertexData;
    vector<unsigned int> indices;
    unsigned int indexCounter = 0;

    for (int y = 0; y < CANVAS_SIZE; y++) {
        for (int x = 0; x < CANVAS_SIZE; x++) {
            int pixelIndex = y * CANVAS_SIZE + x;
            if (canvas[pixelIndex].gray > 0) {
                float pixelX = startX + x * PIXEL_DISPLAY_SIZE;
                float pixelY = startY + y * PIXEL_DISPLAY_SIZE;

                // 直接使用0-255范围的颜色值
                float grayValue = (float)canvas[pixelIndex].gray;

                // 四个顶点
                vertexData.push_back(pixelX);
                vertexData.push_back(pixelY);
                vertexData.push_back(grayValue);

                vertexData.push_back(pixelX + PIXEL_DISPLAY_SIZE);
                vertexData.push_back(pixelY);
                vertexData.push_back(grayValue);

                vertexData.push_back(pixelX + PIXEL_DISPLAY_SIZE);
                vertexData.push_back(pixelY + PIXEL_DISPLAY_SIZE);
                vertexData.push_back(grayValue);

                vertexData.push_back(pixelX);
                vertexData.push_back(pixelY + PIXEL_DISPLAY_SIZE);
                vertexData.push_back(grayValue);

                // 两个三角形
                indices.push_back(indexCounter);
                indices.push_back(indexCounter + 1);
                indices.push_back(indexCounter + 2);
                indices.push_back(indexCounter);
                indices.push_back(indexCounter + 2);
                indices.push_back(indexCounter + 3);
                indexCounter += 4;
            }
        }
    }

    if (!vertexData.empty()) {
        glUseProgram(shaderProgram);
        glUniform2f(glGetUniformLocation(shaderProgram, "screenSize"), (float)screenWidth, (float)screenHeight);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), vertexData.data(), GL_STATIC_DRAW);

        unsigned int EBO;
        glGenBuffers(1, &EBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

        glDeleteBuffers(1, &EBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    // 绘制光标
    drawCursor();
}

// ========== 应用画笔（数据层面加深颜色） ==========
void applyBrush(int centerX, int centerY, float strength, bool isEraser)
{
    int radius = isEraser ? eraserSize : brushSize;

    int startX = max(0, centerX - radius);
    int endX = min(CANVAS_SIZE - 1, centerX + radius);
    int startY = max(0, centerY - radius);
    int endY = min(CANVAS_SIZE - 1, centerY + radius);

    for (int y = startY; y <= endY; y++) {
        for (int x = startX; x <= endX; x++) {
            int dx = x - centerX;
            int dy = y - centerY;
            float dist = sqrt(dx * dx + dy * dy);

            if (dist <= radius) {
                // 距离权重：中心最强，边缘逐渐减弱
                float distanceWeight = 1.0f - (dist / (radius + 1.0f));

                // 【关键修改】大幅增加每次绘制的颜色值
                // 原来是30，现在增加到120，让颜色快速变深
                float increase = strength * distanceWeight * 120.0f;

                int index = y * CANVAS_SIZE + x;

                if (isEraser) {
                    // 擦除：也相应增强
                    canvas[index].gray = max(0, canvas[index].gray - (int)(increase * 1.5f));
                }
                else
                {
                    // 绘制：直接累加大幅增加的颜色值
                    canvas[index].gray = min(255, canvas[index].gray + (int)increase);
                }
            }
        }
    }
}

// ========== 文件操作 ==========
void saveToFile(const string& filename)
{
    ofstream file(filename);
    if (!file.is_open()) {
        cout << "无法创建文件: " << filename << endl;
        return;
    }
    for (int i = 0; i < PIXEL_COUNT; i++) {
        // 直接保存0-255的值
        file << canvas[i].gray << " ";
        if ((i + 1) % 28 == 0)
        {
            file << endl;
        }
    }
    file.close();
    cout << "保存成功: " << filename << " (颜色范围: 0-255)" << endl;
}

void loadFromFile(const string& filename)
{
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "无法打开文件: " << filename << endl;
        return;
    }

    for (int i = 0; i < PIXEL_COUNT; i++) {
        int loadedValue;
        file >> loadedValue;
        // 确保值在0-255范围内
        canvas[i].gray = min(255, max(0, loadedValue));
    }
    file.close();
    cout << "打开成功: " << filename << endl;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT) mousePressed[0] = (action == GLFW_PRESS);
    if (button == GLFW_MOUSE_BUTTON_RIGHT) mousePressed[1] = (action == GLFW_PRESS);
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    mouseX = xpos;
    mouseY = ypos;
    float totalCanvasWidth = CANVAS_SIZE * PIXEL_DISPLAY_SIZE;
    float totalCanvasHeight = CANVAS_SIZE * PIXEL_DISPLAY_SIZE;
    float startX = (screenWidth - totalCanvasWidth) / 2.0f;
    float startY = (screenHeight - totalCanvasHeight) / 2.0f;
    float relativeX = mouseX - startX;
    float relativeY = mouseY - startY;
    isInCanvas = (relativeX >= 0 && relativeX < totalCanvasWidth &&
        relativeY >= 0 && relativeY < totalCanvasHeight);
    if (isInCanvas) {
        int canvasX = (int)(relativeX / PIXEL_DISPLAY_SIZE);
        int canvasY = (int)(relativeY / PIXEL_DISPLAY_SIZE);
        canvasX = max(0, min(CANVAS_SIZE - 1, canvasX));
        canvasY = max(0, min(CANVAS_SIZE - 1, canvasY));
        currentBrushX = canvasX;
        currentBrushY = canvasY;
        if (mousePressed[0]) {
            // 增加绘制强度
            applyBrush(canvasX, canvasY, 2.0f, false);
        }
        if (mousePressed[1]) {
            // 增加擦除强度
            applyBrush(canvasX, canvasY, 2.5f, true);
        }
    }
    else {
        currentBrushX = -1;
        currentBrushY = -1;
    }
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    screenWidth = width;
    screenHeight = height;
    glViewport(0, 0, width, height);
}

void paint()
{
    cout << "==================================================" << endl;
    cout << "                MNIST数字绘图器" << endl;
    cout << "==================================================" << endl;
    cout << "使用方法:" << endl;
    cout << "  鼠标:" << endl;
    cout << "    左键拖动: 绘制 (金色光标) - 颜色更深，一次绘制即可达到深色" << endl;
    cout << "    右键拖动: 擦除 (蓝色光标)" << endl;
    cout << "  控制台命令:" << endl;
    cout << "      1: 保存文件" << endl;
    cout << "      2: 打开文件" << endl;
    cout << "      3: 清空画布" << endl;
    cout << "      4: 显示帮助" << endl;
    cout << "      5: 退出程序" << endl;
    cout << "      6: 设置画笔半径" << endl;
    cout << "      7: 设置橡皮半径" << endl;
    cout << "      8: 切换光标显示" << endl;
    cout << "==================================================" << endl;
    cout << "  金色半透明圆 = 画笔位置和大小" << endl;
    cout << "  蓝色半透明圆 = 橡皮位置和大小" << endl;
    cout << "==================================================" << endl;
    return;
}

void consoleThread()
{
    string filename;
    int ck;

    while (true) {
        cin >> ck;
        switch (ck) {
        case 1:
            cout << "文件保存地址: ";
            if (savefilename.empty()) {
                cin >> filename;
                savefilename = filename;
            }
            else {
                filename = savefilename;
                cout << savefilename << endl;
            }
            saveToFile(filename);
            system("pause");
            system("cls");
            paint();
            break;

        case 2:
            cout << "文件打开地址: ";
            cin >> filename;
            loadFromFile(filename);
            system("pause");
            system("cls");
            paint();
            break;

        case 3:
            for (int i = 0; i < PIXEL_COUNT; i++) {
                canvas[i].gray = 0;
            }
            cout << "画布已清空" << endl;
            system("pause");
            system("cls");
            paint();
            break;

        case 4:
        {
            cout << "\n=== 帮助信息 ===" << endl;
            cout << "1. 保存文件" << endl;
            cout << "2. 打开文件" << endl;
            cout << "3. 清空画布" << endl;
            cout << "4. 显示帮助" << endl;
            cout << "5. 退出程序" << endl;
            cout << "6. 设置画笔半径" << endl;
            cout << "7. 设置橡皮半径" << endl;
            cout << "8. 切换光标显示" << endl;
            cout << "当前画笔半径: " << brushSize << endl;
            cout << "当前橡皮半径: " << eraserSize << endl;
            cout << "光标显示: " << (showBrushCursor ? "开启" : "关闭") << endl;
            cout << "颜色范围: 0-255 (支持深色)" << endl;

            // 显示当前最大颜色值
            int maxColor = 0;
            for (int i = 0; i < PIXEL_COUNT; i++) {
                maxColor = max(maxColor, canvas[i].gray);
            }
            cout << "当前最大颜色值: " << maxColor << endl;
            cout << "===============\n" << endl;
            system("pause");
            system("cls");
            paint();
            break;
        }

        case 5:
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            return;

        case 6:
        {
            cout << "输入新的画笔半径 (1-5): ";
            int newBrushSize;
            cin >> newBrushSize;
            brushSize = max(1, min(5, newBrushSize));
            cout << "画笔半径已设置为: " << brushSize << endl;
            system("pause");
            system("cls");
            paint();
            break;
        }

        case 7:
        {
            cout << "输入新的橡皮半径 (1-8): ";
            int newEraserSize;
            cin >> newEraserSize;
            eraserSize = max(1, min(8, newEraserSize));
            cout << "橡皮半径已设置为: " << eraserSize << endl;
            system("pause");
            system("cls");
            paint();
            break;
        }

        case 8:
            showBrushCursor = !showBrushCursor;
            cout << "光标显示: " << (showBrushCursor ? "开启" : "关闭") << endl;
            system("pause");
            system("cls");
            paint();
            break;

        default:
            cout << "未知命令，输入4查看帮助" << endl;
            system("pause");
            system("cls");
            paint();
            break;
        }
    }
}

int main()
{
    if (!glfwInit()) {
        cout << "GLFW初始化失败" << endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(screenWidth, screenHeight,
        "MNIST绘图器 (28×28 - 数据层面加深颜色)", NULL, NULL);
    if (!window) {
        cout << "窗口创建失败" << endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        cout << "GLAD初始化失败" << endl;
        return -1;
    }

    // 设置视口和回调
    glViewport(0, 0, screenWidth, screenHeight);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    initializeRenderer();
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    paint();
    thread console(consoleThread);

    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        drawPixels();

        glfwSwapBuffers(window);
        glfwPollEvents();
        this_thread::sleep_for(chrono::milliseconds(5));
    }

    console.detach();

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &gridVAO);
    glDeleteBuffers(1, &gridVBO);
    glDeleteVertexArrays(1, &cursorVAO);
    glDeleteBuffers(1, &cursorVBO);
    glDeleteProgram(shaderProgram);
    glDeleteProgram(gridShaderProgram);
    glDeleteProgram(cursorShaderProgram);
    glfwTerminate();

    return 0;
}
