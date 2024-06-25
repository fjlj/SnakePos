// SnakePos.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <stdio.h>
#include <time.h>
#include <Windows.h>
#include <thread>
#include <chrono>
#include <math.h>
#include <random>

#define E 2.71828182845904523536


std::random_device rd;
std::mt19937 gen(rd());
std::mt19937 gener(rd());
std::uniform_int_distribution<int> randm(0, 8);

std::default_random_engine generator;
std::uniform_real_distribution<double> distribution(-2.0, 2.0);
std::uniform_real_distribution<double> oNo(-1.0, 1.0);
std::uniform_real_distribution<double> lR(0.0025, 0.255);
std::uniform_real_distribution<double> zTo(0.0, 1.0);
std::uniform_real_distribution<double> rInps(-3.0, 3.0);
std::uniform_int_distribution<int> sides(1, 19);
std::uniform_int_distribution<int> topb(1, 12);
std::uniform_int_distribution<int> brdm(0, 35165201);

struct InputVars
{
    WORD dir;
    bool running;
    bool drawn;
    bool show;
    double score;
};


class Matrix {

public:
    int rows;
    int cols;
    double** data;


public:
    Matrix(int rows, int cols) {
        this->rows = rows;
        this->cols = cols;
        this->data = new double* [rows * cols];
        memset(this->data, 0.0, sizeof(this->data));

        for (int i = 0; i < this->rows; i++) {
            double* col = new double[this->cols];
            this->data[i] = col;
            for (int j = 0; j < this->cols; j++) {
                this->data[i][j] = 0.0;
            }
        }
    }

    void multiply(double n) {
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                this->data[i][j] *= n;
            }
        }
    }

    void multiply(Matrix& other) {
        if (this->cols != other.cols || this->rows != other.rows) {
            printf("Cannot multiply matricies of differing shape\n");
            return;
        }
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                this->data[i][j] *= other.data[i][j];
            }
        }
    }

    void map(void* n) {
        double (*foo)(double, int, int);
        foo = (double(*)(double, int, int))n;
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                double val = this->data[i][j];
                this->data[i][j] = foo(val, 0, 0);
            }
        }
    }

    static Matrix multiply(const Matrix a, const Matrix other) {
        if (a.cols != other.rows) {
            throw  std::runtime_error("Cols of A must be the same as rows of B.");
        }

        Matrix result(a.rows, other.cols);

        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                double sum = 0.0;
                for (int k = 0; k < a.cols; k++) {
                    sum += a.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    static Matrix map(Matrix a, void* n) {
        Matrix r(a.rows, a.cols);

        double (*foo)(double, int, int);
        foo = (double(*)(double, int, int))n;
        for (int i = 0; i < r.rows; i++) {
            for (int j = 0; j < r.cols; j++) {
                double val = a.data[i][j];
                r.data[i][j] = foo(val, 0, 0);
            }
        }
        return r;
    }

    void add(double n) {
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                this->data[i][j] += n;
            }
        }
    }

    void add(const Matrix& other) {
        if (this->cols != other.cols || this->rows != other.rows) {
            printf("Cannot add matricies of differing shape\n");
            return;
        }
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                this->data[i][j] += other.data[i][j];
            }
        }
    }

    void subtract(const Matrix& other) {
        if (this->cols != other.cols || this->rows != other.rows) {
            printf("Cannot subtract matricies of differing shape\n");
            return;
        }
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                this->data[i][j] -= other.data[i][j];
            }
        }
    }

    static Matrix subtract(const Matrix& a, const Matrix& b) {
        if (a.cols != b.cols || a.rows != b.rows) {
            throw std::runtime_error("Cannot subtract matricies of differing shape\n");
        }
        Matrix result(a.rows, a.cols);
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.cols; j++) {
                result.data[i][j] = a.data[i][j] - b.data[i][j];
            }
        }
        return result;
    }

    static Matrix subtract(const Matrix& a, int b) {
        Matrix result(a.rows, a.cols);
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.cols; j++) {
                result.data[i][j] = a.data[i][j] - b;
            }
        }
        return result;
    }

    void randomize() {
        //srand(time(NULL)+clock());
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                long long someval = (long long)(rand() % 9999) + 1;
                this->data[i][j] = (sin(2 * someval) * cos((5.0 * someval) / 0.01));
                //this->data[i][j] = (double)someval + ((double)rand() / (double)RAND_MAX);
            }
        }
    }

    void fill(double num) {
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                this->data[i][j] = num;
            }
        }
    }

    void copy(const Matrix& other) {
        if (this->cols != other.cols || this->rows != other.rows) {
            printf("Cannot subtract matricies of differing shape\n");
            return;
        }
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                this->data[i][j] = other.data[i][j];
            }
        }
    }

    void transpose() {
        Matrix* t = new Matrix(this->cols, this->rows);
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                t->data[j][i] = this->data[i][j];
            }
        }

        int tmp = this->rows;
        this->rows = this->cols;
        this->cols = tmp;
        for (int k = 0; k < this->cols; k++) {
            delete this->data[k];
            this->data[k] = NULL;
        }
        this->data = NULL;
        *this = *t;
        return;

    }
    static Matrix tpos(Matrix* a) {
        Matrix t(a->cols, a->rows);
        for (int i = 0; i < a->rows; i++) {
            for (int j = 0; j < a->cols; j++) {
                t.data[j][i] = a->data[i][j];
            }
        }
        return t;
    }

    void take_half(const Matrix& other,int s) {
        int which = (rand() % this->rows);
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                if (j % s == 0) {
                    this->data[i][j] = other.data[i][j];
                }
                else {
                    this->data[i][j] = (this->data[i][j] + other.data[i][j] + other.data[i][j]) / 3.0;
                }
            }
        }
    }

    void crossover(const Matrix* other) { //,double max_fit) {
        //srand(time(NULL) + clock());
        //long long int mutate_bits[] = { 0b1111111111111111111111111111111111111111111111111111111111111111, 0b1011111111111111111111111111111111111111111111111111111111111111, 0b1101111111111111111111111111111111111111111111111111111111111111, 0b1110111111111111111111111111111111111111111111111111111111111111, 0b1111011111111111111111111111111111111111111111111111111111111111, 0b1111101111111111111111111111111111111111111111111111111111111111, 0b1111110111111111111111111111111111111111111111111111111111111111, 0b1111111011111111111111111111111111111111111111111111111111111111, 0b1111111101111111111111111111111111111111111111111111111111111111, 0b1111111110111111111111111111111111111111111111111111111111111111, 0b1111111111011111111111111111111111111111111111111111111111111111, 0b1111111111101111111111111111111111111111111111111111111111111111, 0b1111111111110111111111111111111111111111111111111111111111111111, 0b1111111111111011111111111111111111111111111111111111111111111111, 0b1111111111111101111111111111111111111111111111111111111111111111, 0b1111111111111110111111111111111111111111111111111111111111111111, 0b1111111111111111011111111111111111111111111111111111111111111111, 0b1111111111111111101111111111111111111111111111111111111111111111, 0b1111111111111111110111111111111111111111111111111111111111111111, 0b1111111111111111111011111111111111111111111111111111111111111111, 0b1111111111111111111101111111111111111111111111111111111111111111, 0b1111111111111111111110111111111111111111111111111111111111111111, 0b1111111111111111111111011111111111111111111111111111111111111111, 0b1111111111111111111111101111111111111111111111111111111111111111, 0b1111111111111111111111110111111111111111111111111111111111111111, 0b1111111111111111111111111011111111111111111111111111111111111111, 0b1111111111111111111111111101111111111111111111111111111111111111, 0b1111111111111111111111111110111111111111111111111111111111111111, 0b1111111111111111111111111111011111111111111111111111111111111111, 0b1111111111111111111111111111101111111111111111111111111111111111, 0b1111111111111111111111111111110111111111111111111111111111111111, 0b1111111111111111111111111111111011111111111111111111111111111111, 0b1111111111111111111111111111111101111111111111111111111111111111, 0b1111111111111111111111111111111110111111111111111111111111111111, 0b1111111111111111111111111111111111011111111111111111111111111111, 0b1111111111111111111111111111111111101111111111111111111111111111, 0b1111111111111111111111111111111111110111111111111111111111111111, 0b1111111111111111111111111111111111111011111111111111111111111111, 0b1111111111111111111111111111111111111101111111111111111111111111, 0b1111111111111111111111111111111111111110111111111111111111111111, 0b1111111111111111111111111111111111111111011111111111111111111111, 0b1111111111111111111111111111111111111111101111111111111111111111, 0b1111111111111111111111111111111111111111110111111111111111111111, 0b1111111111111111111111111111111111111111111011111111111111111111, 0b1111111111111111111111111111111111111111111101111111111111111111, 0b1111111111111111111111111111111111111111111110111111111111111111, 0b1111111111111111111111111111111111111111111111011111111111111111, 0b1111111111111111111111111111111111111111111111101111111111111111, 0b1111111111111111111111111111111111111111111111110111111111111111, 0b1111111111111111111111111111111111111111111111111011111111111111, 0b1111111111111111111111111111111111111111111111111101111111111111, 0b1111111111111111111111111111111111111111111111111110111111111111, 0b1111111111111111111111111111111111111111111111111111011111111111, 0b1111111111111111111111111111111111111111111111111111101111111111, 0b1111111111111111111111111111111111111111111111111111110111111111, 0b1111111111111111111111111111111111111111111111111111111011111111, 0b1111111111111111111111111111111111111111111111111111111101111111, 0b1111111111111111111111111111111111111111111111111111111110111111, 0b1111111111111111111111111111111111111111111111111111111111011111, 0b1111111111111111111111111111111111111111111111111111111111101111, 0b1111111111111111111111111111111111111111111111111111111111110111, 0b1111111111111111111111111111111111111111111111111111111111111011, 0b1111111111111111111111111111111111111111111111111111111111111101, 0b1111111111111111111111111111111111111111111111111111111111111110 };
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                int numbits = (rand() % 64);
                int mutate_bit = 0;
                long long int m1 = -(((long long int)1) << numbits);
                long long int m2 = ~m1;
                long long int org_shift = (*(long long int*)&this->data[i][j]) & m1;
                long long int new_shift = (*(long long int*)&other->data[i][j]) & m2;
                
                if (((double)rand() / (double)RAND_MAX) < 0.01)
                    mutate_bit = (rand() % 52);
                
                long long int cross_int = ((org_shift | new_shift) ^ ((long long int)1 << mutate_bit));
                //double dec_part = this->data[i][j] - ((int)this->data[i][j]);
                //printf("SHIFT:%d\nORGS:%d\nNEWS:%d\nMERGE:%d\nMUT:%d\n", numbits, org_shift, new_shift, cross_int,mutate_bit);

                    this->data[i][j] =  (*(double*)&cross_int);
            }
        }
    }

    void print() {
        for (int i = 0; i < this->rows; i++)
        {
            for (int j = 0; j < this->cols; j++)
            {
                printf("%f ", this->data[i][j]);
            }
            printf("\n");
        }
    }
};

struct GameVars
{
    bool init;
    double fitness;
    Matrix inputs;

};

double reLu(double num) {
    return max(0, num);
}
double dreLu(double num) {
    return num > 0;
}

double sigmoid(double num) {
    return (1.0 / (1.0 + exp(-num)));
}

double sigmoid2(double num) {
    return (2.0 / (1.0 + exp(-4.9*num))-1);
}

double sign(double num)
{
    return (num < 0 ? 0 : num);
}

double oddone(double num) {
    return (sin(5 * num) * cos(((5*num) / 0.01)));
}

double dsigmoid(double num) {
    return num * (1 - num);
}

float dsigmoid2(float num) {
    return sigmoid2(num) * (1 - sigmoid2(num));
}

float doddone(float num) {
    return oddone(num) * (1 - oddone(num));
}

class NeuralNetwork {

public:
    int iNodes = 0;
    int hNodes = 0;
    int oNodes = 0;
    Matrix* w1, * w2, * b1, * b2 = nullptr;
    unsigned long fitness = 1.0;
    unsigned long long probt = 0;
    uint64_t prob = 0;
    bool bred = false;
    unsigned long long generation = 0;
    float learning_rate = 0.01;
    double mFitpTick = 0;
    BYTE AF1 = 0, AF2 = 0;


public:
    ~NeuralNetwork() {
        this->w1 = nullptr;
        this->w2 = nullptr;
        this->b1 = nullptr;
        this->b2 = nullptr;
        //delete this->w1;

    }

    NeuralNetwork(int i, int h, int o) {
        this->iNodes = i;
        this->hNodes = h;
        this->oNodes = o;
        this->fitness = 1.0;
        this->w1 = new Matrix(h, i);
        this->w1->randomize();
        this->w2 = new Matrix(o, h);
        this->w2->randomize();
        this->b1 = new Matrix(h, 1);
        this->b1->randomize();
        this->b2 = new Matrix(o, 1);
        this->b2->randomize();
        this->bred = false;
        this->generation = 0;
        this->prob = 0.0;
        this->AF1 = 0;
        this->AF2 = 0;
        learning_rate = lR(gen);

    }

    //NeuralNetwork(const NeuralNetwork& n1) {
    //    this->iNodes = n1.iNodes;
    //    this->hNodes = n1.hNodes;
    //    this->oNodes = n1.oNodes;
    //    this->fitness = n1.fitness;
    //    this->learning_rate = n1.learning_rate;
    //    this->w1 = new Matrix(*n1.w1);
    //    this->w2 = new Matrix(*n1.w2);
    //    this->b1 = new Matrix(*n1.b1);
    //    this->b2 = new Matrix(*n1.b2);
    //    this->bred = n1.bred;
    //    this->generation = n1.generation + 1;
    //}

    NeuralNetwork() {
        this->iNodes = 0;
        this->oNodes = 0;
        this->hNodes = 0;
        this->fitness = 1.0;
        this->bred = false;
        this->w1 = nullptr;
        this->w2 = nullptr;
        this->b1 = nullptr;
        this->b2 = nullptr;
        this->generation = 0;
        this->prob = 0.0;
        this->AF1 = 0;
        this->AF2 = 0;
        learning_rate = lR(gen);

    }

    void makeNet(int i, int h, int o) {
        if (this->iNodes != 0) {
        }
        this->iNodes = i;
        this->oNodes = o;
        this->hNodes = h;
        this->w1 = new Matrix(h, i);
        this->w1->randomize();
        this->w2 = new Matrix(o, h);
        this->w2->randomize();
        this->b1 = new Matrix(h, 1);
        this->b1->randomize();
        this->b2 = new Matrix(o, 1);
        this->b2->randomize();
        this->AF1 = 0;
        this->AF2 = 0;
    }

    void copy(const NeuralNetwork& other) {
        this->AF1 = other.AF1;
        this->AF2 = other.AF2;
        this->b1->copy(*other.b1);
        this->b2->copy(*other.b2);
        this->w1->copy(*other.w1);
        this->w2->copy(*other.w2);
        this->hNodes = other.hNodes;
        this->iNodes = other.iNodes;
        this->oNodes = other.oNodes;
        this->fitness = other.fitness;
        this->prob = other.prob;
        this->learning_rate = other.learning_rate;
    }

    Matrix feedForward(Matrix* input)
    {
        //a cols b rows
        Matrix hidden = Matrix::multiply(*this->w1, *input);
        void* afs[] = { &sigmoid, &sigmoid2,&reLu,&oddone };
        void* act1 = afs[this->AF1];
        void* act2 = afs[this->AF2];
        hidden.add(*this->b1);
        hidden.map(act1);


        //a.cols b.rows
        Matrix outputs = Matrix::multiply(*this->w2, hidden);
        
        outputs.add(*this->b2);

        outputs.map(act2);

        return outputs;
    }

    void train(Matrix* inputs, Matrix* targets) {
        void* afs[] = { &sigmoid, &sigmoid2, &reLu,&oddone };
        void* derivs[] = { &dsigmoid, &dsigmoid2, &dreLu,&doddone };
        Matrix hidden = Matrix::multiply(*this->w1, *inputs);


        hidden.map(afs[this->AF1]);

        Matrix outputs = Matrix::multiply(*this->w2, hidden);
        outputs.map(afs[this->AF2]);

        Matrix o_error = Matrix::subtract(*targets, outputs);

        Matrix who_t = Matrix::tpos(this->w2);

        Matrix h_error = Matrix::multiply(who_t, o_error);

        Matrix grad = Matrix::map(outputs, derivs[this->AF1]);
        grad.multiply(o_error);
        grad.multiply(this->learning_rate);

        Matrix h_grad = Matrix::map(hidden, derivs[this->AF2]);
        h_grad.multiply(h_error);
        h_grad.multiply(this->learning_rate);

        Matrix hidden_outputs_T = Matrix::tpos(&hidden);
        Matrix weights_ho_deltas = Matrix::multiply(grad, hidden_outputs_T);
        this->w2->add(weights_ho_deltas);


        Matrix wih_t = Matrix::tpos(inputs);
        Matrix weights_ih_deltas = Matrix::multiply(h_grad, wih_t);
        this->w1->add(weights_ih_deltas);


        //Matrix hidden_t = Matrix::tpos(this->w2);




        /*printf("\nout:\n");
        outputs.print();
        printf("\nerror:\n");
        o_error.print();
        printf("\n");*/
    }
};


double randMToN(double M, double N)
{
    return M + (rand() / (RAND_MAX / (N - M)));
}


int dirIndex(NeuralNetwork* brain, Matrix* inputs, int cDir)
{
    int inv_dirs[] = { 1, 0, 3, 2, 
                       0, 1, 2, 3 };
    //srand(time(NULL) + clock());
    double max = 0;
    int max_ind = 0;
    int second_max_ind = 0;
    int choice_ind = inv_dirs[cDir];
    int attempts = 10;

    while ((choice_ind == inv_dirs[cDir] || cDir == choice_ind) && attempts>0) {
        Matrix outs = brain->feedForward(inputs);
        for (int i = 0; i < outs.rows; i++) {
            if (outs.data[i][0] > max) {
                max = outs.data[i][0];
                second_max_ind = max_ind % 4;
                max_ind = i % 4;
            }
            else {
                if (outs.data[i][0] > 0.0000) {
                    max_ind = (double)rand() / (double)RAND_MAX >= 0.99 ? i % 4: max_ind;
                    //break;
                }
                else {
                    if (!isnormal(outs.data[i][0])) {
                        outs.data[i][0] = ((double)rand() / (double)RAND_MAX) - 0.5;
                    }
                    else {
                        outs.data[i][0] += ((double)(rand() / 4.0) / (double)RAND_MAX) - 0.125;
                    }
                }
            }
        }
        if (inv_dirs[max_ind] != cDir && cDir != inv_dirs[max_ind+4]) {
            choice_ind = inv_dirs[max_ind];
        }
        else if (inv_dirs[second_max_ind] != cDir && cDir != inv_dirs[second_max_ind+4]){
            choice_ind = inv_dirs[second_max_ind];
        }
        else {
            Matrix targets(4, 1);
            targets.data[cDir][0] = 0.05;
            targets.data[inv_dirs[cDir]][0] = 0.05;
            targets.data[(cDir + (2 - (cDir % 2))) % 4][0] = 0.83;
            targets.data[(cDir + (3 - (cDir % 2))) % 4][0] = 0.83;

            brain->train(inputs, &targets);
            if (--attempts <= 0)
                choice_ind = inv_dirs[(cDir + ((rand() % 2 + 2) - (cDir % 2))) % 4];
        }
        //attempts--;
    }
    return inv_dirs[choice_ind];
}

void getDirKey(InputVars* iVars) {
    WORD inv_dirs[] = { 1, 0, 3, 2, 0, 1, 2, 3};
    int check_keys[] = { 0x57 ,0x53,0x41,0x44,VK_UP,VK_DOWN,VK_LEFT,VK_RIGHT};

    int itr = 0;
    while ((iVars->running || iVars->dir == 4) /* || true*/) {
        if (iVars->drawn || iVars->dir == 4) {
            while (itr < 8) {
                if (((GetAsyncKeyState(check_keys[itr]) & 0x01) || (GetAsyncKeyState(check_keys[itr]) < 0)) && iVars->dir != inv_dirs[itr % 4] && iVars->dir != inv_dirs[(itr % 4) + 4]) {
                    if (iVars->dir == 4 && !iVars->drawn) {
                        return;
                    }
                    else {
                        iVars->dir = inv_dirs[(itr % 4) + 4];
                        iVars->drawn = false;
                        break;
                    }
                }
                itr++;
            }
            itr = 0;  
        }

        if ((GetAsyncKeyState(VK_ESCAPE) & 0x01)) {
            iVars->running = false;
            iVars->dir = 0;
        }
    }
    return;
}

int rand_food_pos(int width, int height, int border_tb, int border_lr) {
    int f_x = (rand() % (width-(border_lr*2))) + border_lr;
    int f_y = (rand() % (height-(border_tb*2))) + border_tb;
    return width * f_y + f_x;
}

void print_segs(int* segs, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d", segs[i]);
        if (i != 0 && i % 50 == 0) printf("\n");
    }
    printf("\n");
}

double tail(int* segments, int size) {
    int pos = 0;
    for (int i = 0; i < size; i++) {
        if (segments[i] == 1) {
            pos = i;
            break;
        };
    }
    return (double)pos;
}

double mapSum(char* map, size_t size, int height) {
    int width = (size / sizeof(char)) / height;
    unsigned long long int tot = 0;
    int *ret = nullptr;
    unsigned char sums[8] = {1,2,3,4,5,6,7,8};
    for (int j = 0; j < width; ++j) {
        for (int i = 0; i < height; ++i) {
            int ind = i * width + j;
            if (map[ind] == 32) continue;
                sums[i%8] += sums[j%8] + map[ind];
                sums[j%8] += sums[i%8] + map[ind];
                
        }
    }
    tot = (*((size_t*)&sums));
    //printf("%lf\n", tot / 18446744073709551615.0);
    //system("pause");
    return tot/18446744073709551615.0;
}

void game(InputVars *iVars, Matrix *gVars, int gen, double avg_fit, double max_fit)
{

    HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);

    CONSOLE_CURSOR_INFO info;
    info.dwSize = 100;
    info.bVisible = FALSE;

    COORD org;
    org.X = 0;
    org.Y = 0;

    DWORD dwMode;
    GetConsoleMode(hStdout, &dwMode);
    dwMode |= ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hStdout, dwMode);

    CONSOLE_FONT_INFOEX cfi;
    cfi.cbSize = sizeof(cfi);
    cfi.nFont = 0;
    cfi.dwFontSize.X = 20;                   // Width of each character in the font
    cfi.dwFontSize.Y = 20;                  // Height
    cfi.FontFamily = FF_DONTCARE;
    cfi.FontWeight = FW_BOLD;
    wcscpy_s(cfi.FaceName, L"Cascadia Mono SemiBold"); // Choose your font

    SMALL_RECT rect;
    COORD coord;
    coord.X = 40; // Defining our X and
    coord.Y = 30;  // Y size for buffer.

    rect.Top = 0;
    rect.Left = 0;
    rect.Bottom = coord.Y - 1; // height for window
    rect.Right = coord.X - 1;  // width for window

    SetCurrentConsoleFontEx(hStdout, TRUE, &cfi);
    SetConsoleScreenBufferSize(hStdout, coord);       // set buffer size
    SetConsoleWindowInfo(hStdout, TRUE, &rect);
    ShowScrollBar(GetConsoleWindow(), SB_BOTH, 0);
    SetConsoleCursorInfo(hStdout, &info);


    char red[] = "\033[91m"; //12
    char color_off[] = "\033[00m"; //5
    char green[] = "\033[92m";
    char blue[] = "\033[95m";
    char yellow[] = "\033[93m";

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

    int s_segments[sizeof(map) / sizeof(char)];
    memset(s_segments, 0, sizeof(s_segments));

    int height = 26;
    int width = (sizeof(map) / sizeof(char)) / height;

    int dirs[5] = { width * -1,width,-1,1,0 };
    int segments = 1;
    int speed_calc;
    std::thread::id m_id = std::this_thread::get_id();
    //srand((unsigned int)time(NULL) + rand() + (int)(int*)&m_id);
    srand(123456789);
    int s_h_pos = (width * ((rand() % (int)(height * 0.4)) + (height * 0.2))) + ((rand() % (int)(width * 0.4)) + (width * 0.2));
    int f_pos = rand_food_pos(width, height, 1, 2);
    int tail_pos = 0;

    bool exit = false;
    unsigned long long apple_timer = 100, life_timer = 0, run_score = 0, score = 0;
    int hunger = 1000;

    //clock_t start_t, end_t;
    int speed = (int)(180.0l * (0.851l * (1.01l - ((double)segments / 165.01l))));

    map[s_h_pos] = '*';
    map[f_pos] = '@';

    //int last_dir = iVars->dir;

    while (!exit) {
        
        s_h_pos += dirs[iVars->dir]; 
        s_segments[s_h_pos] = segments;
        

        for (int i = 0; i < sizeof(s_segments) / sizeof(int); i++) {
            if (s_segments[i] > 0) {

                if (map[i] == 'x' || (map[i] == '*' && s_h_pos == i && iVars->dir != 4) || iVars->running == false) {
                    exit = true;
                    //iVars->score /= 10;
                    iVars->running = false;
                }
                if (map[i] == '@') {
                    segments += 1;
                    score += apple_timer * 5;
                    apple_timer = 100;
                    hunger = 1000;
                    s_segments[i]++;
                    map[i] = '*';
                    f_pos = rand_food_pos(width, height, 1, 2);
                    while (s_segments[f_pos] > 0 || map[f_pos] != ' ') { // || map[f_pos] == '\n' || map[f_pos] == '\t') {
                        f_pos = rand_food_pos(width, height, 1, 2);
                    }
                    map[f_pos] = '@';

                }
                map[i] = '*';
                s_segments[i]--;
            }
            else if (map[i] == '*') {
                map[i] = ' ';
            }
        }
        SetConsoleCursorPosition(hStdout, org);

        if (exit) {
            printf("%sGAME OVER%s%s", red, (iVars->running == true ? " - ESC = EXIT, OTHER = PLAY" : " - Thank You For Playing!"), color_off);
        }

        speed_calc = (int)((1.0l - ((180.0l * (0.851l * (1.01l - ((double)segments / 165.01l)))) / 153.18l)) * 100);
        
        if(iVars->dir != 4) score += ((life_timer % 3)/2); //* ((double)speed_calc + (last_dir == iVars->dir ? 0 : 0.5)));
        
        run_score = score+(20*((double)segments-1));

        //gVars->data[0][0] = iVars->dir;
        gVars->data[1][0] = mapSum(map, sizeof(map), height);
        //gVars->data[0][0] = segSum(s_segments, sizeof(s_segments) / sizeof(s_segments[0]));

        

        printf("\n SEGMENTS:%-6dSCORE:%-7lluSPEED:% 3d%%\n", segments, run_score, speed_calc);
        if (iVars->score - max_fit >= 10 && avg_fit > 0.0) {
            printf("%s", map);
            std::this_thread::sleep_for(std::chrono::milliseconds((int)speed / 3));
        } else {
            if (avg_fit == -1) {
                printf("%s", blue);

                for (int o = 0; o < (sizeof(map) / sizeof(map[0])); o++) {
                    if (map[o] == '@') {
                        printf("%s@%s", green, blue);
                    }
                    else if (map[o] == '*') {
                        if (map[o - 1] != '*')
                            printf("%s", yellow);

                        printf("*");

                        if (map[o + 1] != '*')
                            printf("%s", blue);
                    }
                    else {
                        printf("%c", map[o]);
                    }
                }
                printf("%s", color_off);
                std::this_thread::sleep_for(std::chrono::milliseconds((int)speed / 2));
            }
        }
        printf("%sDIR:%d GEN:%d MAP:%lf", color_off,iVars->dir,gen, gVars->data[1][0]);
        speed = (int)(180.0l * (0.851l * (1.01l - ((double)segments / 165.01l))));
        //std::this_thread::sleep_for(std::chrono::milliseconds(speed));
        //last_dir = iVars->dir;

        if (apple_timer > 1) {
            apple_timer--;
        }
       else {
            hunger--;
            if(hunger <= 0){
                exit = true;
                iVars->running = false;
            }
        }

        //tail_pos = tail(s_segments, (sizeof(s_segments) / sizeof(s_segments[0])));
        gVars->data[0][0] = f_pos;
        //gVars->data[1][0] = f_pos;
        gVars->data[2][0] = (double)abs((s_h_pos % width) - (f_pos % width));
        gVars->data[3][0] = (double)abs((s_h_pos / height) - (f_pos / height));
        //gVars->data[4][0] = (double)(s_h_pos % width);
        //gVars->data[5][0] = (double)(s_h_pos / height);
        //gVars->data[6][0] = (double)(f_pos % width);
        //gVars->data[7][0] = (double)(f_pos / height);
        //gVars->data[8][0] = (double)tail_pos;
        //gVars->data[9][0] = (double)(tail_pos / height);
        //gVars->data[10][0] = (double)(tail_pos % width);
        iVars->score = run_score;

        iVars->drawn = true;
        life_timer++;
    }
    iVars->drawn = false;
    iVars->dir = 4;
}

void runNet(NeuralNetwork* brain, Matrix* inputs, InputVars* iVars ) {
    while (iVars->running) {
        if (iVars->drawn && &inputs != nullptr) {
            iVars->dir = dirIndex(brain,inputs, iVars->dir);
            iVars->drawn = false;
        }
    }
    return;

}

void breedBrains(NeuralNetwork* brains, int num_brains,double avg_fit, double max_fit,int gen) {
    //srand(time(NULL) + clock());
    int num_bred = 0;
    int max_fit_ind = 0;
    double max_fit_seen = 0.0;
    int max_fit_breed_cnt = num_brains * 0.05;
    int high_fits = 0;
    for (int i = 0; i < num_brains; i++) {
        if (brains[i].fitness > max_fit_seen) {
            max_fit_seen = brains[i].fitness;
            max_fit_ind = i;
        }
    }
    for (int i = 0; i < num_brains; i++) {
        if (max_fit_seen - brains[i].fitness <= 10) {
            high_fits++;
        }
    }
        for (int i = 0; i < num_brains; i++) {
            //if (brains[i].fitness == max_fit) max_fit_ind = i;
            if (brains[i].bred) continue;
                for (int j = 0; j < num_brains; j++) {
                    //if (max_fit_seen - brains[j].fitness <= 50 && max_fit_seen > 40) continue;
                    if (!brains[j].bred && i != j && (brains[j].fitness * 0.97) <= brains[i].fitness && (((double)rand() / (double)RAND_MAX) <= 0.56 || (max_fit_seen - brains[i].fitness <= 10 && max_fit_breed_cnt > 0 && i != j && (((double)rand() / (double)RAND_MAX) <= 0.87)))) {//|| (!brains[j].bred && j != i && brains[j].fitness < brains[i].fitness && ((double)rand() / (double)RAND_MAX) >= (((brains[j].fitness + brains[i].fitness) + 1) / ((max_fit + 1) * 1.33)) && (((double)rand() / (double)RAND_MAX) - 0.75 >= (double)num_bred / num_brains))) {

                        if (max_fit_seen - brains[i].fitness <= 10) {
                            if (--max_fit_breed_cnt < 0) {
                                brains[i].bred = true;
                                num_bred++;
                                brains[i].generation++;
                                if (--high_fits > 0) {
                                    max_fit_breed_cnt = num_brains * 0.05;
                                }
                            }
                        } else {
                                brains[i].bred = true;
                        }
                                brains[j].bred = true;
                                brains[j].generation++;
                                num_bred++;
                                brains[j].w1->take_half(*brains[i].w1, rand() % 2 + 1);
                                brains[j].w2->take_half(*brains[i].w2, rand() % 2 + 1);
                                break;
                    }
                            //num_bred++;
                }
                        

                        /*brains[j].w1->crossover(brains[i].w1, max_fit);
                        brains[j].w2->crossover(brains[i].w2, max_fit);*/
                        //take_half
                        //brains[j].generation++;
                        //brains[i].generation++;
                        //break;
                    //}
                    
        }
        int mutated = 0;
        for (int u = 0; u < num_brains; u++) {
            brains[u].bred = false;
            //if (brains[u].fitness - avg_fit<= 5) continue;

            if (gen % 9 == 0 && max_fit_seen - brains[u].fitness <= 10 ) {
                brains[u].w1->crossover(brains[u].w1);
                brains[u].w2->crossover(brains[u].w2);
                brains[u].generation = 0;
                mutated++;
            }
        }

        printf("GEN:%d\nNUM BRED:%d\n MAX FIT:%f\n AVG FIT:%f\nMUTATES:%d", gen,num_bred,max_fit_seen,avg_fit,mutated);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
}

void doAI() {
    //define variable structures for controlling the game and receiving game data
    InputVars* iVars = new InputVars;
    memset(iVars, 0, sizeof(InputVars));
    Matrix* gVars = new Matrix(4, 1);

    //define variables that will be used in creating the networks
    const int pop_size = 200;
    const int generations = 100000;
    double avg_fit = 0.0;
    double fitness_max = 0.0;
    int max_fit_brain = 0;

    //build a network or networks
    NeuralNetwork* brains = (NeuralNetwork*)malloc(sizeof(NeuralNetwork[pop_size]));
    if (brains == nullptr) return;
    memset(brains, 0, sizeof(NeuralNetwork[pop_size]));
    for (int i = 0; i < pop_size; i++) {
        brains[i].makeNet(4, 6, 4);
    }

    //run each network and score them
#pragma loop(hint_parallel( 4 ))
    for (int gen = 0; gen < generations; gen++) {
        iVars->running = true;
        for (int j = 0; j < pop_size; j++) {


            while (iVars->running) {
                //initialize starting game state
                iVars->dir = 4;
                iVars->running = true;
                iVars->drawn = false;

                //launch the game in its own thread (could do more than one for each network)
                std::thread teh_game(game, iVars, gVars, gen, avg_fit, fitness_max);
                //launch the current network
                std::thread think(runNet, &brains[j], gVars, iVars);

                //wait for them to finish
                teh_game.join();
                think.join();
                teh_game.~thread();
                think.~thread();

            }
            //set the fitness score of the network
            brains[j].fitness = iVars->score;
            system("cls");
            
            //prompt starting the next network
            iVars->running = true;
        }

        //calculate max, and averages
        double fitness_sum = 0.0;
        //double had_fit = 0.0;
        double local_max = 0.0;

        for (int k = 0; k < pop_size; k++) {
            if (brains[k].fitness > 0.0) {
                fitness_sum += brains[k].fitness;
                //had_fit++;
            }
            if (brains[k].fitness > local_max) {
                local_max = brains[k].fitness;
                max_fit_brain = k;
            }
        }

        if (local_max > fitness_max)
            fitness_max = local_max;

        avg_fit = fitness_sum / (double)pop_size;

        //breed population
        breedBrains(brains, pop_size, avg_fit, local_max, gen);

    }
    //cleanup
    delete iVars;
    delete gVars;
    free(brains);
}

void play() {
    InputVars* iVars = new InputVars;
    Matrix* gVars = new Matrix(11, 1);

    iVars->running = true;

    while (iVars->running) {
        iVars->dir = 4;
        iVars->running = true;
        iVars->drawn = true;

        std::thread teh_game(game, iVars, gVars, 0, -1,0);
        std::thread input(getDirKey, iVars);

        teh_game.join();
        input.join();
    }


    delete iVars;
    delete gVars;
}

int main() {
    //while (1) {
    //play();
//}

    doAI();
    system("pause");
    return 0;
}

