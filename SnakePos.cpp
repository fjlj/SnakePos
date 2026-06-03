#include <vector>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <chrono>
#include <random>
#include <execution>
#include <fstream>
#include <string>
#include <filesystem>
#include <sstream>
#include <ctime>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")

#define TINYMATRIX_IMPLEMENTATION
#include "TinyMatrix.h"

// =========================================================
// 1. ENGINE CONSTANTS & STATE
// =========================================================
#define GAUNTLET_SEED_FETCH (*(volatile unsigned int*)0x7FFE0014)

constexpr int MAX_GENERATIONS = 10000;
constexpr int MAX_VISION_RAY = 4;
constexpr int MAX_POP_SIZE = 2000;
constexpr int INPUT_NODES = 13;
constexpr int OUTPUT_NODES = 4;
constexpr int MAX_HIDDEN_NODES = (INPUT_NODES * 2) + (INPUT_NODES / 3);
constexpr int MAX_PATIENCE = 600;
const unsigned int GAUNTLET_SEED = GAUNTLET_SEED_FETCH;
constexpr int INITIAL_POP_SIZE = 150;
constexpr int MAX_POP_CYCLES = 10;

std::atomic<bool> keep_running(true);
std::atomic<bool> is_exhibition_running(false);
std::atomic<bool> skip_exhibition(false);
std::atomic<bool> reset_exhibition(false);
std::atomic<bool> hide_exhibition(true);
std::atomic<bool> warp_speed(false);
std::atomic<bool> goto_best(false);
std::atomic<bool> game_beaten(false);

std::mutex hof_mutex;
std::mutex dashboard_mutex;
std::string cached_dashboard = "";

struct EngineState {
    int pop_size = INITIAL_POP_SIZE;
    int hidden_nodes = INPUT_NODES;
    int patience = MAX_PATIENCE;
    int stagnant_gens = 0;
    double all_time_best = 0.0;
    double epoch_best = 0.0;
    double next_milestone = 3000.0;
    int pop_cycles = 0;
};

struct PopStats {
    double avg_fitness;
    double max_fitness;
    double convergence;
    int max_wins;
};

// =========================================================
// 2. THE BRAIN
// =========================================================
class SnakeBrain {
public:
    TinyMatrix W1, W2, B1, B2;
    TinyMatrix hidden_buf;
    TinyMatrix output_buf;
    double fitness = 0.0;
    int wins = 0;

    SnakeBrain() = default;

    SnakeBrain(int inputs, int hidden, int outputs) {
        W1.Reserve(inputs, MAX_HIDDEN_NODES);
        W2.Reserve(MAX_HIDDEN_NODES, outputs);
        B1.Reserve(1, MAX_HIDDEN_NODES);
        B2.Reserve(1, outputs);

        W1.Shape(inputs, hidden).Floats().Randomize(-1.0f, 1.0f);
        W2.Shape(hidden, outputs).Floats().Randomize(-1.0f, 1.0f);
        B1.Shape(1, hidden).Floats().Randomize(-1.0f, 1.0f);
        B2.Shape(1, outputs).Floats().Randomize(-1.0f, 1.0f);

        hidden_buf.Shape(1, MAX_HIDDEN_NODES);
        output_buf.Shape(1, outputs);
    }

    TinyMatrix& feedForward(TinyMatrix& inputs) {
        hidden_buf.dot(inputs, W1).add(B1).Tanh();
        output_buf.dot(hidden_buf, W2).add(B2).Sigmoid();
        return output_buf; 
    }

    void serialize(int generation, const std::string& filename) {
        std::ofstream out(filename);
        if(!out.is_open()) return;

        out << "// ==========================================\n";
        out << "// NEAT CHAMPION BRAIN\n";
        out << "// Generation : " << generation << "\n";
        out << "// Fitness    : " << (long long)this->fitness << "\n";
        out << "// ==========================================\n\n";

        auto dumpMatrix = [&](TinyMatrix& m, const std::string& name) {
            out << "const float " << name << "[" << m.Rows() << "][" << m.Cols() << "] = {\n";
            for(int i = 0; i < m.Rows(); i++) {
                out << "    {";
                for(int j = 0; j < m.Cols(); j++) {
                    out << m(i, j) << (j == m.Cols() - 1 ? "" : ", ");
                }
                out << "}" << (i == m.Rows() - 1 ? "" : ",") << "\n";
            }
            out << "};\n\n";
            };

        dumpMatrix(W1, "CHAMPION_W1");
        dumpMatrix(B1, "CHAMPION_B1");
        dumpMatrix(W2, "CHAMPION_W2");
        dumpMatrix(B2, "CHAMPION_B2");

        out.close();
    }

    void complexify(int new_hidden) {
        if(new_hidden <= W1.Cols()) return;

        W1.SetLogicalBounds(W1.Rows(), new_hidden);
        B1.SetLogicalBounds(1, new_hidden);
        W2.SetLogicalBounds(new_hidden, W2.Cols());
    }

    void crossover(SnakeBrain& parentA, SnakeBrain& parentB, float mutation_rate = 0.05f) {
        thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> style_picker(0, 2);

        int style = style_picker(rng);

        crossMatrix(this->W1, parentA.W1, parentB.W1, mutation_rate, style);
        crossMatrix(this->B1, parentA.B1, parentB.B1, mutation_rate, style);
        
        style = style_picker(rng);
        crossMatrix(this->W2, parentA.W2, parentB.W2, mutation_rate, style);
        crossMatrix(this->B2, parentA.B2, parentB.B2, mutation_rate, style);
    }

    void prune(float limit = 5.0f) {
        auto clampLambda = [limit](float val, int r, int c) {
            return std::clamp(val, -limit, limit);
            };

        this->W1.mapInline(clampLambda);
        this->W2.mapInline(clampLambda);
        this->B1.mapInline(clampLambda);
        this->B2.mapInline(clampLambda);
    }

private:
    void crossMatrix(TinyMatrix& target, TinyMatrix& a, TinyMatrix& b, float m_rate, int breed_style) {
        thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> coin_flip(0, 1);
        std::uniform_real_distribution<float> chance(0.0f, 1.0f);
        std::uniform_real_distribution<float> rat(-0.50f, 0.50f);
        std::uniform_real_distribution<float> mutation_shift(-0.2f, 0.2f);

        for(int r = 0; r < target.Rows(); r++) {
            bool use_parent_A_for_row = (coin_flip(rng) == 0);

            for(int c = 0; c < target.Cols(); c++) {
                float val = 0.0f;

                switch(breed_style) {
                case 0: // Uniform
                    val = (coin_flip(rng) == 0) ? a(r, c) : b(r, c);
                    break;
                case 1: // Whole-Row Splicing
                    val = use_parent_A_for_row ? a(r, c) : b(r, c);
                    break;
                case 2: // Arithmetic Blending
                {
                    float ratio = rat(rng);
                    val = (a(r, c) * ratio) + (b(r, c) * (1.0f - ratio));
                    break;
                }
                }

                if(chance(rng) < m_rate) val += mutation_shift(rng);
                target(r, c, val);
            }
        }
    }
};

std::vector<SnakeBrain> champion_queue;
SnakeBrain absolute_best_brain;

struct FastRNG {
    uint32_t state;

    FastRNG(uint32_t seed) {
        state = (seed == 0) ? GAUNTLET_SEED_FETCH : seed;
    }

    inline uint32_t operator()() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }
};

// =========================================================
// 3. PHYSICS & EVALUATION ENGINE
// =========================================================
double evaluateBrain(SnakeBrain& brain, bool draw, unsigned int seed, int play_current = 0, int play_total = 0,bool replay_mode = false) {
    FastRNG rng(seed);

    const int height = 15;
    const int width = 20;
    const int MAP_SIZE = height * width;

    char map[] = "xxxxxxxxxxxxxxxxxxxx"
        "x                  x"
        "x                  x"
        "x                  x"
        "x                  x"
        "x                  x"
        "x                  x"
        "x                  x"
        "x                  x"
        "x                  x"
        "x                  x"
        "x                  x"
        "x                  x"
        "x                  x"
        "xxxxxxxxxxxxxxxxxxxx";

    int body[MAP_SIZE];
    int b_head = 0;
    int b_tail = 0;

    int dirs[4] = {width * -1, width, -1, 1};
    int segments = 1;

    auto getMaxHunger = [](int segs) { return 150 + (segs * 2); };
    int hunger = getMaxHunger(segments);
    double score = 0;

    int s_h_pos = (width * (height / 2)) + (width / 2);
    int f_pos = 0;
    bool exit = false;
    int current_dir = 3;

    auto spawn_apple = [&]() -> bool {
        int empty_spots[MAP_SIZE];
        int empty_count = 0;

        for(int i = 0; i < MAP_SIZE; i++) {
            if(map[i] == ' ') {
                empty_spots[empty_count++] = i;
            }
        }

        if(empty_count == 0) return false; // WIN CONDITION

        f_pos = empty_spots[rng() % empty_count];
        map[f_pos] = '@';
        return true;
        };

    map[s_h_pos] = '*';
    body[b_head++] = s_h_pos;
    spawn_apple();

    thread_local TinyMatrix vision(1, INPUT_NODES);

    while(!exit && keep_running) {
        int h_x = s_h_pos % width;
        int h_y = s_h_pos / width;
        int f_x = f_pos % width;
        int f_y = f_pos / width;

        vision(0, 0, (float)(f_x - h_x) / (float)width);
        vision(0, 1, (float)(f_y - h_y) / (float)height);

        int ray_offsets[8] = {
            dirs[0], dirs[1], dirs[2], dirs[3],
            dirs[0] - 1, dirs[0] + 1,
            dirs[1] - 1, dirs[1] + 1
        };

        auto castRay = [&](int start_pos, int offset) {
            int current_pos = start_pos;
            for(int d = 1; d <= MAX_VISION_RAY; d++) {
                current_pos += offset;
                if(current_pos < 0 || current_pos >= MAP_SIZE) return 1.0f / (float)d;

                char tile = map[current_pos];
                if(tile == 'x' || tile == '*' || tile == '\n') {
                    return 1.0f / (float)d;
                }
            }
            return 0.0f;
            };

        for(int i = 0; i < 8; i++) {
            vision(0, 2 + i, castRay(s_h_pos, ray_offsets[i]));
        }

        vision(0, 10, (current_dir == 2) ? -1.0f : (current_dir == 3) ? 1.0f : 0.0f);
        vision(0, 11, (current_dir == 0) ? -1.0f : (current_dir == 1) ? 1.0f : 0.0f);
        vision(0, 12, (float)hunger / (float)getMaxHunger(segments));

        TinyMatrix& outs = brain.feedForward(vision);

        float max_confidence = outs(0, current_dir);
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
        bool ate_food = (s_h_pos == f_pos);

        if(!ate_food) {
            int old_tail = body[b_tail];
            map[old_tail] = ' ';
            b_tail = (b_tail + 1) % MAP_SIZE;
        }

        if(s_h_pos < 0 || s_h_pos >= MAP_SIZE || map[s_h_pos] == 'x' || map[s_h_pos] == '*') {
            exit = true;
        }

        if(!exit) {
            body[b_head] = s_h_pos;
            b_head = (b_head + 1) % MAP_SIZE;
            map[s_h_pos] = '*';

            if(ate_food) {
                double capacity = (double)segments / (double)MAP_SIZE;
                double speed_mult = 0.0;
                if(capacity < 0.66) {
                    double normalized_cap = capacity / 0.66;
                    speed_mult = 1.0 - (normalized_cap * normalized_cap);
                }
                double speed_bonus = ((double)hunger / getMaxHunger(segments)) * 3000.0 * speed_mult;
                score += 2000 + speed_bonus;
                segments++;
                hunger = getMaxHunger(segments);

                if(!spawn_apple()) {
                    brain.wins += 1;
                    exit = true;
                }
            }
        }

        hunger--;
        if(hunger <= 0) exit = true;
        score += 1.0f;

        if(draw) {
            while(!replay_mode && hide_exhibition && !skip_exhibition && keep_running) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }

            if(skip_exhibition) {
                skip_exhibition = false;
                exit = true;
                break;
            }

            double live_fitness = score + ((double)(segments * segments * segments) * 100.0);

            char frame_buffer[8192];
            int len = 0;
            len += snprintf(frame_buffer + len, sizeof(frame_buffer) - len,
                "\033[1;1H"
                "  \033[00m--- CHAMPION EXHIBITION REPLAY ---   \n"
                "  SEG: %-3d | FIT: %-10lld | #: %d/%d    \n\033[95m",
                segments, (long long)live_fitness, play_current, play_total);

            enum ColorState {
                BLUE, GREEN, YELLOW
            };
            ColorState current_color = BLUE;

            auto change_color = [&](ColorState target, const char* code, int code_len) {
                if(current_color != target) {
                    memcpy(frame_buffer + len, code, code_len);
                    len += code_len;
                    current_color = target;
                }
                };

            for(int o = 0; o < MAP_SIZE; o++) {
                if(o > 0 && o % width == 0) frame_buffer[len++] = '\n';
                if(o % width == 0) frame_buffer[len++] = ' ';

                if(map[o] == '@') {
                    change_color(GREEN, "\033[92m", 5);
                    frame_buffer[len++] = '@';
                } else if(map[o] == '*') {
                    change_color(YELLOW, "\033[93m", 5);
                    frame_buffer[len++] = '*';
                } else {
                    change_color(BLUE, "\033[95m", 5);
                    frame_buffer[len++] = map[o];
                }
            }
            memcpy(frame_buffer + len, "\033[00m", 5);
            len += 5;
            frame_buffer[len] = '\0';

            fputs(frame_buffer, stdout);

            if(warp_speed) {
                std::this_thread::sleep_for(std::chrono::milliseconds(6));
            } else {
                int frame_delay = (int)(11.0 + (49.0 * std::exp(-0.0125 * segments)));
                std::this_thread::sleep_for(std::chrono::milliseconds(frame_delay));
            }
        }
    }

    double length_bonus = (double)(segments * segments * segments) * 100.0;
    return score + length_bonus;
}

// =========================================================
// 4. ASYNC THREADS (Input & Ghost)
// =========================================================
void inputThread() {
    bool esc_was_down = false;
    bool tab_was_down = false;
    bool b_was_down = false;

    while(keep_running) {
        bool esc_is_down = (GetAsyncKeyState(VK_ESCAPE) & 0x8000) != 0;
        bool ctrl_is_down = (GetAsyncKeyState(VK_CONTROL) & 0x8000) != 0;
        warp_speed = ctrl_is_down;

        if(esc_is_down && !esc_was_down) {
            if(is_exhibition_running) skip_exhibition = true;
            else if(!champion_queue.empty()) reset_exhibition = true;
        }
        esc_was_down = esc_is_down;

        bool tab_is_down = (GetAsyncKeyState(VK_TAB) & 0x8000) != 0;
        if(tab_is_down && !tab_was_down) {
            hide_exhibition = !hide_exhibition;
            printf("\033[2J");
            if(hide_exhibition) {
                std::lock_guard<std::mutex> lock(dashboard_mutex);
                printf("%s", cached_dashboard.c_str());
            }
        }
        tab_was_down = tab_is_down;

        bool b_is_down = (GetAsyncKeyState('B') & 0x8000) != 0;
        if(b_is_down && !b_was_down) {
            goto_best = true;
            if(is_exhibition_running) skip_exhibition = true;
        }
        b_was_down = b_is_down;

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void ghostThread() {
    int playback_index = 0;

    while(keep_running) {
        if(goto_best) {
            std::lock_guard<std::mutex> lock(hof_mutex);
            if(!champion_queue.empty()) {
                playback_index = (int)champion_queue.size() - 1;
            }
            goto_best = false;
            reset_exhibition = false;
            is_exhibition_running = true;
        }

        if(reset_exhibition) {
            playback_index = 0;
            reset_exhibition = false;
            is_exhibition_running = true;
        }

        SnakeBrain champ_to_play;
        bool available = false;
        int current_play = 0;
        int total_in_queue = 0;

        {
            std::lock_guard<std::mutex> lock(hof_mutex);
            if(playback_index < (int)champion_queue.size()) {
                champ_to_play = champion_queue[playback_index];
                current_play = playback_index + 1;
                total_in_queue = (int)champion_queue.size();
                playback_index++;
                available = true;
            }
        }

        if(available) {
            is_exhibition_running = true;
            printf("\033[2J");

            if(champ_to_play.W1.Rows() > 0) {
                evaluateBrain(champ_to_play, true, GAUNTLET_SEED_FETCH, current_play, total_in_queue);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(555));
            if(!hide_exhibition) printf("\033[2J");

            if(current_play == total_in_queue) is_exhibition_running = false;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
}

// =========================================================
// 5. ENGINE ORCHESTRATION HELPER FUNCTIONS
// =========================================================
void logEvent(const std::string& run_folder, int gen, const PopStats& stats, const EngineState& state, const std::string& event_type) {
    std::ofstream log(run_folder + "/telemetry.log", std::ios::app);
    if(log.is_open()) {
        char buffer[512];
        snprintf(buffer, sizeof(buffer),
            "[GEN %-5d] %-25s | Pop: %-4d | Nodes: %-3d | Conv: %-4.2f | Wins: %-2d | Max: %-10.0f | Avg: %-10.0f\n",
            gen, event_type.c_str(), state.pop_size, state.hidden_nodes,
            stats.convergence, stats.max_wins, state.all_time_best, stats.avg_fitness); 

        log << buffer;
        log.close();
    }
}

void printDashboard(int gen, const PopStats& stats, const EngineState& state) {
    char buffer[1024];
    snprintf(buffer, sizeof(buffer),
        "\033[1;1H"
        "--- NEAT ENGINE DIAGNOSTICS ---\n"
        "Generation     : %-6d        \n"
        "All-Time Best  : %-10.0f    \n"
        "Gen Best Fit   : %-10.0f    \n"
        "Pop Avg Fit    : %-10.0f    \n"
        "Convergence    : %-4f        \n"
        "Stagnation     : %-4d/%-4d  \n"
        "Nodes/Pop      : %-3d/%-4d  \n"
        "-------------------------------\n",
        gen, state.all_time_best, stats.max_fitness, stats.avg_fitness,
        stats.convergence, state.stagnant_gens, state.patience, state.hidden_nodes, state.pop_size);

    {
        std::lock_guard<std::mutex> lock(dashboard_mutex);
        cached_dashboard = buffer;
    }

    if(!is_exhibition_running || hide_exhibition) {
        printf("%s", buffer);
    }
}

void printEventAlert(const std::string& message) {
    if(!is_exhibition_running || hide_exhibition) {
        printf("\033[11;1H\n===================================\n%s===================================\n", message.c_str());
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        printf("\033[11;1H\033[2K\033[12;1H\033[2K\033[13;1H\033[2K\033[14;1H\033[2K\033[15;1H\033[2K\033[16;1H\033[2K");
    }
}

PopStats calculateStats(const std::vector<SnakeBrain>& population) {
    double sum_fitness = 0.0;
    int max_wins = 0;
    for(const auto& brain : population) {
        sum_fitness += brain.fitness;
        if(brain.wins > max_wins) {
            max_wins = brain.wins; 
        }
    }

    double max_fitness = population.front().fitness;
    double avg_fitness = sum_fitness / population.size();
    double convergence = (max_fitness > 0) ? (avg_fitness / max_fitness) : 0.0;

    return {avg_fitness, max_fitness, convergence, max_wins};
}

void saveGeneBank(std::vector<SnakeBrain>& population) {
    std::ofstream out("gene_bank.bin", std::ios::binary);
    if(!out) return;

    std::sort(population.begin(), population.end(), [](const SnakeBrain& a, const SnakeBrain& b) {
        return a.fitness > b.fitness;
        });

    int num_to_save = std::min((int)population.size(), 10);
    out.write((char*)&num_to_save, sizeof(int));

    int inputs = INPUT_NODES;
    int outputs = OUTPUT_NODES;
    out.write((char*)&inputs, sizeof(int));
    out.write((char*)&outputs, sizeof(int));

    for(int i = 0; i < num_to_save; i++) {
        auto& brain = population[i];

        int hidden = brain.W1.Cols();
        out.write((char*)&hidden, sizeof(int));

        brain.W1.WriteRaw(out);
        brain.B1.WriteRaw(out);
        brain.W2.WriteRaw(out);
        brain.B2.WriteRaw(out);
    }
    out.close();
}

bool injectGeneBank(std::vector<SnakeBrain>& population, EngineState& state) {
    std::ifstream in("gene_bank.bin", std::ios::binary);
    if(!in) return false;

    int num_saved, saved_inputs, saved_outputs;
    in.read((char*)&num_saved, sizeof(int));
    in.read((char*)&saved_inputs, sizeof(int));
    in.read((char*)&saved_outputs, sizeof(int));

    if(saved_inputs != INPUT_NODES || saved_outputs != OUTPUT_NODES) {
        printf("\n\033[91m[ERROR] Base architecture changed! Cannot load Gene Bank.\033[00m\n");
        return false;
    }

    int max_hidden = INPUT_NODES;

    for(int i = 0; i < num_saved && i < population.size(); i++) {
        int hidden;
        in.read((char*)&hidden, sizeof(int));
        max_hidden = std::max(max_hidden, hidden);

        population[i] = SnakeBrain(INPUT_NODES, hidden, OUTPUT_NODES);

        population[i].W1.ReadRaw(in);
        population[i].B1.ReadRaw(in);
        population[i].W2.ReadRaw(in);
        population[i].B2.ReadRaw(in);

        population[i].fitness = 999999999.0;
    }

    state.hidden_nodes = max_hidden;
    for(auto& brain : population) {
        brain.complexify(max_hidden);
    }

    in.close();
    return true;
}

void checkHallOfFame(std::vector<SnakeBrain>& population, EngineState& state, int gen, const std::string& run_folder) {
    if(population[0].fitness > state.epoch_best) {
        state.epoch_best = population[0].fitness;
        state.stagnant_gens = 0;
    } else {
        state.stagnant_gens++;
    }

    if(population[0].fitness > state.all_time_best) {
        state.all_time_best = population[0].fitness;
        absolute_best_brain = population[0];

        if(state.all_time_best >= state.next_milestone) {
            char filename[512];
            snprintf(filename, sizeof(filename), "%s/champion_gen_%d_fit_%lld.h",
                run_folder.c_str(), gen, (long long)state.all_time_best);
            population[0].serialize(gen, filename);

            {
                std::lock_guard<std::mutex> lock(hof_mutex);
                champion_queue.push_back(population[0]);
            }
            saveGeneBank(population);
            logEvent(run_folder, gen, calculateStats(population), state, "NEW MILESTONE REACHED - SYSTEM: AUTOSAVE");
            state.next_milestone = std::max(state.all_time_best * 1.10, state.all_time_best + 5000.0);
        }
    } 
}

void triggerPhaseShift(std::vector<SnakeBrain>& population, EngineState& state, int gen, const std::string& run_folder, const PopStats& stats) {
    if(state.stagnant_gens >= state.patience) {

        bool can_grow_nodes = (state.hidden_nodes < MAX_HIDDEN_NODES);
        bool can_grow_pop = (state.pop_size < MAX_POP_SIZE);
        bool grew_nodes = false;
        bool grew_pop = false;

        if(can_grow_nodes || can_grow_pop) {
            if(state.pop_cycles >= MAX_POP_CYCLES && can_grow_nodes) {
                grew_nodes = true;
                state.pop_cycles = 0;
            }
            else if(can_grow_pop) {
                grew_pop = true;
                state.pop_cycles++;
            }
            else if(can_grow_nodes) {
                grew_nodes = true;
                state.pop_cycles = 0;
            }
        }

        if(grew_nodes) {
            state.hidden_nodes++;
            for(auto& brain : population) brain.complexify(state.hidden_nodes);
        } else if(grew_pop) {
            state.pop_size = std::min(state.pop_size * 2, MAX_POP_SIZE);
        }

        population.resize(state.pop_size, SnakeBrain(INPUT_NODES, state.hidden_nodes, OUTPUT_NODES));
        state.stagnant_gens = 0;
        state.epoch_best = 0.0;

        if(grew_nodes || grew_pop) {
            std::string event_name = grew_nodes ? "SHIFT: ADDED NODES" : "SHIFT: POP EXPANDED";
            logEvent(run_folder, gen, stats, state, event_name);

            char buffer[256];
            snprintf(buffer, sizeof(buffer),
                ">> PHASE SHIFT! STAGNATION       <<\n"
                ">> SHIFT TYPE: %17s <<\n"
                ">> METRICS: %d POP / %d NODES   <<\n",
                grew_nodes ? "NEURAL COMPLEXIFY" : "POP EXPANSION", state.pop_size, state.hidden_nodes);
            printEventAlert(buffer);
        }
    }
}

void performReproduction(std::vector<SnakeBrain>& population, const EngineState& state) {
    int elites = (int)(state.pop_size * 0.15);
    thread_local std::mt19937 rng(std::random_device{}());

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    auto get_biased_elite = [&]() {
        float r = dist(rng);
        return (int)(r * r * elites);
        };

    float stag_ratio = (float)state.stagnant_gens / (float)state.patience;

    for(int i = elites; i < state.pop_size; i++) {
        int p1 = get_biased_elite();
        int p2 = get_biased_elite();

        float parent_avg = (population[p1].fitness + population[p2].fitness) / 2.0f;
        float fitness_ratio = (population[0].fitness > 0) ? (parent_avg / population[0].fitness) : 0.05f;

        // 1. Total parameters in the network (L)
        float L = (float)((INPUT_NODES * state.hidden_nodes) + (state.hidden_nodes * OUTPUT_NODES));

        // 2. The 1/L Dynamic Baseline
        float RATE_BASE = 1.0f / L;

        const float RATE_FLOOR = RATE_BASE / 1.5f;
        const float RATE_CEIL = RATE_BASE * 3.0f; 

        float final_rate = 0.0f;

        if(population[0].fitness < 9000.0f) {
            final_rate = RATE_CEIL * 1.5f;
        } else {
            float inverse_ratio = 1.0f - fitness_ratio;
            float base_curve = inverse_ratio * inverse_ratio;

            float current_ceil = RATE_CEIL + (RATE_CEIL * stag_ratio);
            float current_floor = RATE_FLOOR + (RATE_FLOOR * (stag_ratio * 0.5f));

            float dynamic_rate = current_floor + ((current_ceil - current_floor) * base_curve);
            final_rate = std::clamp(dynamic_rate, current_floor, current_ceil);
        }

        population[i].crossover(population[p1], population[p2], final_rate);
        population[i].prune();
    }
}

// =========================================================
// 6. MAIN LOOP
// =========================================================
void doAI() {
    auto t = std::time(nullptr);
    struct tm tm_info;
    localtime_s(&tm_info, &t);
    std::ostringstream folder_stream;
    folder_stream << "runs/run_" << std::put_time(&tm_info, "%Y%m%d_%H%M%S");
    std::string run_folder = folder_stream.str();
    std::filesystem::create_directories(run_folder);

    auto engine_start_time = std::chrono::high_resolution_clock::now();

    EngineState state;
    std::vector<SnakeBrain> population(state.pop_size, SnakeBrain(INPUT_NODES, state.hidden_nodes, OUTPUT_NODES));

    std::thread inputListener(inputThread);
    std::thread visualizer(ghostThread);

    printf("\033[2J");

    int gen = 0;

    if(injectGeneBank(population, state)) {
        std::sort(population.begin(), population.end(), [](const SnakeBrain& a, const SnakeBrain& b) {
            return a.fitness > b.fitness;
            });

        absolute_best_brain = population[0];

        {
            std::lock_guard<std::mutex> lock(hof_mutex);
            champion_queue.push_back(population[0]);
        }

        logEvent(run_folder, gen, calculateStats(population), state, "SYSTEM: GENE BANK LOADED");
        printf("\n\033[92m[SYSTEM] GENE BANK INJECTED! Seeding Gen 0 with Legendary DNA...\033[00m\n");
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    for(; keep_running && gen < MAX_GENERATIONS; gen++) {

        std::for_each(std::execution::par, population.begin(), population.end(), [gen](SnakeBrain& brain) {
            double score1 = std::max(0.0, evaluateBrain(brain, false, GAUNTLET_SEED + gen));
            double score2 = std::max(0.0, evaluateBrain(brain, false, GAUNTLET_SEED + gen + 100));
            double score3 = std::max(0.0, evaluateBrain(brain, false, GAUNTLET_SEED + gen + 200));

            brain.fitness = std::sqrt(score1) * std::sqrt(score2) * std::sqrt(score3);
            });

        std::sort(population.begin(), population.end(), [](const SnakeBrain& a, const SnakeBrain& b) {
            return a.fitness > b.fitness;
            });

        PopStats stats = calculateStats(population);
        
        if(population[0].wins >= 2) {
            game_beaten = true;
            auto engine_end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = engine_end_time - engine_start_time;
            int hrs = (int)elapsed.count() / 3600;
            int mins = ((int)elapsed.count() % 3600) / 60;
            double secs = elapsed.count() - (hrs * 3600) - (mins * 60);

            char time_msg[128];
            snprintf(time_msg, sizeof(time_msg), "CONQUERED IN %02dh:%02dm:%05.2fs", hrs, mins, secs);
            logEvent(run_folder, gen, stats, state, time_msg);
           
            std::this_thread::sleep_for(std::chrono::milliseconds(500));

            printf("\033[2J\033[1;1H");
            printf("\033[92m");
            printf("===============================================================\n");
            printf(">>  SINGULARITY ACHIEVED : PERFECT HAMILTONIAN CYCLE FOUND   <<\n");
            printf(">>          THE AI HAS OFFICIALLY BEATEN THE GAME            <<\n");
            printf(">>                                                           <<\n");
            printf(">>  TIME TO SINGULARITY      : %02dh:%02dm:%05.2fs           <<\n", hrs, mins, secs);
            printf("===============================================================\n");
            printf("\033[00m\n");

            saveGeneBank(population);
            keep_running = false;
            break;
        }

        checkHallOfFame(population, state, gen, run_folder);
        triggerPhaseShift(population, state, gen, run_folder, stats);

        // ==========================================
        // MICRO-CULLING: GEOMETRIC GRAVITY DECAY
        // ==========================================
        int larger_of = (INITIAL_POP_SIZE > state.pop_size ? INITIAL_POP_SIZE : state.pop_size);
        int decay_floor = INITIAL_POP_SIZE + (larger_of / 4);
        int overpop = state.pop_size - decay_floor;

        if(overpop > 0) {
            // Geometric tiering: 2 -> 4 -> 8 -> 16 -> 32 
            int cull_delay = 32;              
            if(overpop > 64) cull_delay = 2;  
            else if(overpop > 32) cull_delay = 4;  
            else if(overpop > 16) cull_delay = 8;  
            else if(overpop > 8)  cull_delay = 16; 

            if(gen % cull_delay == 0) {
                state.pop_size--;
                population.pop_back();
            }
        }
        // ==========================================

        printDashboard(gen, stats, state);
        performReproduction(population, state);
    }

    keep_running = false;
    inputListener.join();
    visualizer.join();

    if(!(game_beaten)) {
        auto engine_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = engine_end_time - engine_start_time;
        int hrs = (int)elapsed.count() / 3600;
        int mins = ((int)elapsed.count() % 3600) / 60;
        double secs = elapsed.count() - (hrs * 3600) - (mins * 60);

        char time_msg[128];
        bool hit_max_gens = (gen >= MAX_GENERATIONS);

        if(hit_max_gens) {
            snprintf(time_msg, sizeof(time_msg), "MAX GENS REACHED: %02dh:%02dm:%05.2fs", hrs, mins, secs);
        } else {
            snprintf(time_msg, sizeof(time_msg), "USER TERMINATED: %02dh:%02dm:%05.2fs", hrs, mins, secs);
        }

        PopStats final_stats = calculateStats(population);
        logEvent(run_folder, gen, final_stats, state, time_msg);

        printf("\033[93m\n==========================================\n");
        if(hit_max_gens) {
            printf(">> RUN TERMINATED: MAX GENS REACHED     <<\n");
        } else {
            printf(">> RUN TERMINATED BY USER               <<\n");
        }
        printf(">> TOTAL RUNTIME: %02dh:%02dm:%05.2fs       <<\n", hrs, mins, secs);
        printf("==========================================\033[00m\n");
    }

    printf("\n\033[93m[SYSTEM] Committing final memory dump to disk...\033[00m\n");
    saveGeneBank(population);

    if(state.all_time_best > 0) {
        char filename[512];
        snprintf(filename, sizeof(filename), "%s/champion_FINAL_gen_%d_fit_%lld.h",
            run_folder.c_str(), gen, (long long)state.all_time_best);
        absolute_best_brain.serialize(gen, filename);

        printf("\033[11;1H\n==========================================\n");
        printf("[SYSTEM] Saved Final Champion to: \n%s\n", filename);
        printf("==========================================\n\n");
    }

    TinyMatrix::CleanupEngine();
}

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
        printf("\n[SYSTEM] Received Ctrl-C! Initiating graceful shutdown...\n");
        keep_running = false;
        return TRUE;
    }
    default:
        return FALSE;
    }
}
// =========================================================
// 7. INFINITE REPLAY MODE
// =========================================================
void doReplay() {
    std::ifstream in("gene_bank_replay.bin", std::ios::binary);
    if(!in) {
        printf("\033[91m[ERROR] Could not open gene_bank_replay.bin\033[00m\n");
        return;
    }

    int num_saved, saved_inputs, saved_outputs;
    in.read((char*)&num_saved, sizeof(int));
    in.read((char*)&saved_inputs, sizeof(int));
    in.read((char*)&saved_outputs, sizeof(int));

    if(saved_inputs != INPUT_NODES || saved_outputs != OUTPUT_NODES) {
        printf("\n\033[91m[ERROR] Base architecture changed! Cannot load Replay Bank.\033[00m\n");
        return;
    }

    std::vector<SnakeBrain> replay_brains;
    int max_hidden = INPUT_NODES;
    int numLoaded = std::min(5, num_saved);

    for(int i = 0; i < numLoaded; i++) {
        int hidden;
        in.read((char*)&hidden, sizeof(int));
        max_hidden = std::max(max_hidden, hidden);

        SnakeBrain brain(INPUT_NODES, hidden, OUTPUT_NODES);
        brain.W1.ReadRaw(in);
        brain.B1.ReadRaw(in);
        brain.W2.ReadRaw(in);
        brain.B2.ReadRaw(in);
        replay_brains.push_back(brain);
    }
    in.close();

    // Ensure all brains are matched to the maximum architecture size
    for(auto& brain : replay_brains) {
        brain.complexify(max_hidden);
    }

    printf("\033[2J\033[1;1H");
    printf("\n\033[96m ======================================\n");
    printf(" > SYSTEM: REPLAY MODE ACTIVATED      <\n");
    printf(" > Loaded %d Brain(s) (Max Nodes: %-2d) <\n", num_saved, max_hidden);
    printf(" ======================================\033[00m\n");
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Spin up the input thread so hotkeys (CTRL, TAB, ESC) still work!
    std::thread inputListener(inputThread);

    int match_count = 1;
    int total_matches = numLoaded * 3; // 3 random maps per brain

    while(keep_running) {
        for(int i = 0; i < numLoaded && keep_running; i++) {

            // Play 5 completely random seeds per brain
            for(int s = 0; s < 3 && keep_running; s++) {
                unsigned int random_seed = GAUNTLET_SEED_FETCH;
                game_beaten = false; // Reset the global flag so it doesn't break

                printf("\033[2J");
                evaluateBrain(replay_brains[i], true, random_seed, match_count++, total_matches, true);

                // If the user hit ESC to skip, reset the flag for the next map
                if(skip_exhibition) skip_exhibition = false;

                std::this_thread::sleep_for(std::chrono::milliseconds(2500));
            }
        }
        match_count = 1;
    }

    keep_running = false;
    inputListener.join();
}

int main() {
    timeBeginPeriod(1);
    SetConsoleCtrlHandler(CtrlHandler, TRUE);
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

    if(std::filesystem::exists("gene_bank_replay.bin")) {
        doReplay();
    } else {
        doAI();
    }
    timeEndPeriod(1);
    return 0;
}