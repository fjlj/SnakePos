#include <iostream>
#include <vector>
#include <algorithm>
#include <time.h>
#include <Windows.h>
#include <chrono>
#include <thread>
#include <atomic>
#include <random>
#include <mutex>

#define TINYMATRIX_IMPLEMENTATION
#include "TinyMatrix.h"

// ---------------------------------------------------------
// The Brain
// ---------------------------------------------------------
class SnakeBrain {
public:
    TinyMatrix W1, W2, B1, B2;
    double fitness = 0.0;

    SnakeBrain() {
    }
    SnakeBrain(int inputs, int hidden, int outputs) {
        W1.Shape(inputs, hidden).Randomize(-1.0f, 1.0f);
        W2.Shape(hidden, outputs).Randomize(-1.0f, 1.0f);
        B1.Shape(1, hidden).Randomize(-1.0f, 1.0f);
        B2.Shape(1, outputs).Randomize(-1.0f, 1.0f);
    }

    TinyMatrix feedForward(TinyMatrix& inputs) {
        TinyMatrix hidden(1, W1.Cols());
        hidden.dot(inputs, W1).add(B1).Tanh();

        TinyMatrix outputs(1, W2.Cols());
        outputs.dot(hidden, W2).add(B2).Sigmoid();

        return outputs;
    }

    void crossover(SnakeBrain& parentA, SnakeBrain& parentB, float mutation_rate = 0.05f) {
        crossMatrix(this->W1, parentA.W1, parentB.W1, mutation_rate);
        crossMatrix(this->W2, parentA.W2, parentB.W2, mutation_rate);
        crossMatrix(this->B1, parentA.B1, parentB.B1, mutation_rate);
        crossMatrix(this->B2, parentA.B2, parentB.B2, mutation_rate);
    }

private:
    void crossMatrix(TinyMatrix& target, TinyMatrix& a, TinyMatrix& b, float m_rate) {
        for(int r = 0; r < target.Rows(); r++) {
            for(int c = 0; c < target.Cols(); c++) {
                float val = (rand() % 2 == 0) ? a(r, c) : b(r, c);
                if(((float)rand() / (float)RAND_MAX) < m_rate) {
                    val += (((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f) * 0.2f;
                }
                target(r, c, val);
            }
        }
    }
};

// ---------------------------------------------------------
// Globals & Ctrl-C Handler
// ---------------------------------------------------------
std::atomic<bool> keep_running(true);
std::atomic<bool> trigger_exhibition(false);
std::atomic<bool> is_exhibition_running(false);
std::mutex hof_mutex;
SnakeBrain hof_brain_copy;
const unsigned int GAUNTLET_SEED = 1337;

BOOL WINAPI CtrlHandler(DWORD fdwCtrlType) {
    switch(fdwCtrlType) {
    case CTRL_C_EVENT:
    case CTRL_CLOSE_EVENT:
    {
        HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
        CONSOLE_CURSOR_INFO info;
        info.dwSize = 100;
        info.bVisible = TRUE;
        SetConsoleCursorInfo(hStdout, &info);
        printf("\033[0m\n");

        keep_running = false;
        TinyMatrix::CleanupEngine();
        return FALSE;
    }
    default:
        return FALSE;
    }
}

// ---------------------------------------------------------
// Game Execution Engine
// ---------------------------------------------------------

double evaluateBrain(SnakeBrain& brain, bool draw, unsigned int seed) {
    std::mt19937 rng(seed);

    auto rand_food_pos = [&](int width, int height, int border_tb, int border_lr) {
        int f_x = (rng() % (width - (border_lr * 2))) + border_lr;
        int f_y = (rng() % (height - (border_tb * 2))) + border_tb;
        return width * f_y + f_x;
        };
    char map[] = " xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
        " x                                    x\n"
        " x                                    x\n"
        " x                                    x\n"
        " x                                    x\n"
        " x                                    x\n"
        " x                                    x\n"
        " x                                    x\n"
        " x                                    x\n"
        " x                                    x\n"
        " x                                    x\n"
        " x                                    x\n"
        " x                                    x\n"
        " x                                    x\n"
        " x                                    x\n"
        " x                                    x\n"
        " x                                    x\n"
        " x                                    x\n"
        " x                                    x\n"
        " x                                    x\n"
        " x                                    x\n"
        " x                                    x\n"
        " x                                    x\n"
        " x                                    x\n"
        " x                                    x\n"
        " xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n";

    int height = 26;
    int width = 40;
    int s_segments[1040] = {0};

    int dirs[4] = {width * -1, width, -1, 1};
    int segments = 1;

    auto getMaxHunger = [](int segs) { return 150 + (segs * 2); };

    int hunger = getMaxHunger(segments);
    double score = 0;

    int s_h_pos = (width * (height / 2)) + (width / 2);
    int f_pos = rand_food_pos(width, height, 1, 2);

    bool exit = false;
    int current_dir = 3;

    map[f_pos] = '@';

    HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
    COORD org = {0, 0};

    while(!exit) {
        TinyMatrix vision(1, 6);
        int h_x = s_h_pos % width;
        int h_y = s_h_pos / width;
        int f_x = f_pos % width;
        int f_y = f_pos / width;

        vision(0, 0, (float)(f_x - h_x) / (float)width);
        vision(0, 1, (float)(f_y - h_y) / (float)height);

        auto isDanger = [&](int pos) {
            if(pos < 0 || pos >= sizeof(map)) return true;
            if(map[pos] == 'x' || map[pos] == '*' || map[pos] == '\n') return true;
            return false;
            };

        vision(0, 2, isDanger(s_h_pos + dirs[0]) ? 1.0f : 0.0f);
        vision(0, 3, isDanger(s_h_pos + dirs[1]) ? 1.0f : 0.0f);
        vision(0, 4, isDanger(s_h_pos + dirs[2]) ? 1.0f : 0.0f);
        vision(0, 5, isDanger(s_h_pos + dirs[3]) ? 1.0f : 0.0f);

        TinyMatrix outs = brain.feedForward(vision);
        float max_confidence = -9999.0f;
        int best_dir = current_dir;

        for(int i = 0; i < 4; i++) {
            if(outs(0, i) > max_confidence) {
                max_confidence = outs(0, i);
                best_dir = i;
            }
        }

        if((current_dir == 0 && best_dir != 1) || (current_dir == 1 && best_dir != 0) ||
            (current_dir == 2 && best_dir != 3) || (current_dir == 3 && best_dir != 2)) {
            current_dir = best_dir;
        }

        s_h_pos += dirs[current_dir];
        s_segments[s_h_pos] = segments;

        for(int i = 0; i < sizeof(s_segments) / sizeof(int); i++) {
            if(s_segments[i] > 0) {
                if(map[i] == 'x' || (map[i] == '*' && s_h_pos == i)) exit = true;
                if(map[i] == '@') {
                    int current_max = getMaxHunger(segments);
                    double speed_bonus = ((double)hunger / current_max) * 100.0;
                    score += 100 + speed_bonus;

                    segments++;

                    hunger = getMaxHunger(segments);

                    s_segments[i]++;
                    map[i] = '*';

                    f_pos = rand_food_pos(width, height, 1, 2);
                    while(s_segments[f_pos] > 0 || map[f_pos] != ' ') {
                        f_pos = rand_food_pos(width, height, 1, 2);
                    }
                    map[f_pos] = '@';
                }
                map[i] = '*';
                s_segments[i]--;
            } else if(map[i] == '*') {
                map[i] = ' ';
            }
        }

        hunger--;
        if(hunger <= 0) exit = true;
        score += 1;

        // --- RENDER BLOCK ---
        if(draw) {
            char color_off[] = "\033[00m";
            char green[] = "\033[92m";
            char blue[] = "\033[95m";
            char yellow[] = "\033[93m";
            std::string frame_buffer;
            frame_buffer.reserve(4096);
            double live_fitness = score + (std::pow(segments, 3) * 100.0);
            char header[256];
            snprintf(header, sizeof(header),
                "   %s--- CHAMPION EXHIBITION MATCH ---   \n"
                "   SEG: %-3d | FIT: %-7d | ENG: %-3d \n",
                color_off, segments, (int)live_fitness, hunger);
            frame_buffer += header;

            frame_buffer += blue;
            for(int o = 0; o < (sizeof(map) / sizeof(map[0])) - 1; o++) {
                if(map[o] == '@') {
                    frame_buffer += green;
                    frame_buffer += "@";
                    frame_buffer += blue;
                } else if(map[o] == '*') {
                    if(o > 0 && map[o - 1] != '*') frame_buffer += yellow;
                    frame_buffer += "*";
                    if(o < sizeof(map) - 1 && map[o + 1] != '*') frame_buffer += blue;
                } else {
                    frame_buffer += map[o];
                }
            }
            frame_buffer += color_off;

            SetConsoleCursorPosition(hStdout, org);
            printf("%s", frame_buffer.c_str());

            int frame_delay = (int)(15.0 + (65.0 * std::exp(-0.01 * segments)));
            std::this_thread::sleep_for(std::chrono::milliseconds(frame_delay));
        }
    }

    double length_bonus = std::pow(segments, 3) * 100.0;

    return score + length_bonus;
}

// ---------------------------------------------------------
// 4. Async Threads
// ---------------------------------------------------------

void inputThread() {
    bool space_was_down = false;
    while(keep_running) {
        bool space_is_down = (GetAsyncKeyState(VK_SPACE) & 0x8000) != 0;

        if(space_is_down && !space_was_down && !is_exhibition_running) {
            is_exhibition_running = true;
        }
        space_was_down = space_is_down;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

// GHOST RENDERER
void ghostThread() {
    HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
    COORD top_left = {0, 0};

    while(keep_running) {
        if(is_exhibition_running) {
            SnakeBrain local_champ;
            {
                std::lock_guard<std::mutex> lock(hof_mutex);
                local_champ = hof_brain_copy;
            }

            SetConsoleCursorPosition(hStdout, top_left);
            for(int i = 0; i < 30; i++) printf("                                        \n");

            if(local_champ.W1.Rows() > 0) {
                evaluateBrain(local_champ, true, GAUNTLET_SEED);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(555));
            SetConsoleCursorPosition(hStdout, top_left);
            for(int i = 0; i < 30; i++) printf("                                        \n");

            is_exhibition_running = false;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
}

// ---------------------------------------------------------
// Genetic Algorithm
// ---------------------------------------------------------
void doAI() {
    srand((unsigned int)time(NULL));

    const int POP_SIZE = 800;
    const int GENERATIONS = 5000;

    std::vector<SnakeBrain> population(POP_SIZE, SnakeBrain(6, 18, 4));

    double all_time_best_fitness = 0.0;
    double next_milestone = 100.0;

    // Launch background threads
    std::thread inputListener(inputThread);
    std::thread visualizer(ghostThread);

    HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
    COORD top_left = {0, 0};

    for(int gen = 0; gen < GENERATIONS; gen++) {

        for(int i = 0; i < POP_SIZE; i++) {
            population[i].fitness = evaluateBrain(population[i], false, GAUNTLET_SEED);

            if(population[i].fitness > all_time_best_fitness) {
                all_time_best_fitness = population[i].fitness;

                {
                    std::lock_guard<std::mutex> lock(hof_mutex);
                    hof_brain_copy = population[i];
                }

                if(all_time_best_fitness >= next_milestone && !is_exhibition_running) {
                    is_exhibition_running = true;
                    next_milestone = all_time_best_fitness * 1.10;
                }
            }
        }

        std::sort(population.begin(), population.end(), [](const SnakeBrain& a, const SnakeBrain& b) {
            return a.fitness > b.fitness;
            });

        if(!is_exhibition_running) {
            SetConsoleCursorPosition(hStdout, top_left);
            printf("--- NEAT ENGINE TRAINING ---\n");
            printf("Generation     : %-6d        \n", gen);
            printf("Gen Best Fit   : %-6.0f      \n", population[0].fitness);
            printf("All-Time Best  : %-6.0f      \n", all_time_best_fitness);
            printf("\n[Press SPACEBAR to see the Champion]\n");
        }

        int elites = (int)(POP_SIZE * 0.15);
        for(int i = elites; i < POP_SIZE; i++) {
            int p1 = rand() % elites;
            int p2 = rand() % elites;
            population[i].crossover(population[p1], population[p2], 0.05f);
        }
    }

    keep_running = false;
    inputListener.join();
    visualizer.join();
    TinyMatrix::CleanupEngine();
}

int main() {
    SetConsoleCtrlHandler(CtrlHandler, TRUE); // Ctrl-C teardown handler

    HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);

    DWORD dwMode;
    GetConsoleMode(hStdout, &dwMode);
    dwMode |= ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hStdout, dwMode);

    CONSOLE_FONT_INFOEX cfi;
    cfi.cbSize = sizeof(cfi);
    cfi.nFont = 0;
    cfi.dwFontSize.X = 20;
    cfi.dwFontSize.Y = 20;
    cfi.FontFamily = FF_DONTCARE;
    cfi.FontWeight = FW_BOLD;
    wcscpy_s(cfi.FaceName, L"Cascadia Mono SemiBold");
    SetCurrentConsoleFontEx(hStdout, TRUE, &cfi);

    // --- The Console Resize Dance ---
    SMALL_RECT minRect = {0, 0, 1, 1};
    SetConsoleWindowInfo(hStdout, TRUE, &minRect);
    COORD coord = {40, 31};
    SetConsoleScreenBufferSize(hStdout, coord);
    SMALL_RECT rect = {0, 0, 39, 30};
    SetConsoleWindowInfo(hStdout, TRUE, &rect);

    CONSOLE_CURSOR_INFO info;
    info.dwSize = 100;
    info.bVisible = FALSE;
    SetConsoleCursorInfo(hStdout, &info);

    doAI();
    return 0;
}