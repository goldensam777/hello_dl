#ifndef MICROGRAD_H
#define MICROGRAD_H

#include <stddef.h>

#define VALUE_CHILDREN_COUNT 2

typedef struct Value Value;
typedef Value value;

typedef struct {
    Value **values;
    size_t size;
} ValueArray;

struct Value {
    double data;
    double grad;
    double exponent;
    char op;
    const char *label;
    Value *children[VALUE_CHILDREN_COUNT];
};

typedef struct Neuron {
    Value **weights;
    size_t n_in;
    Value *bias;
    ValueArray params;
} Neuron;

typedef struct Layer {
    Neuron *neurons;
    size_t n_in;
    size_t n_out;
    ValueArray params;
} Layer;

typedef struct MLP {
    Layer *layers;
    size_t n_layers;
    ValueArray params;
} MLP;

Value *init_value(double data_, double grad_, char op_, const char *label_,
                 Value **children_);
void print_value(const value *val);
Value *val_add(Value *v1, Value *v2);
Value *val_mul(Value *v1, Value *v2);
Value *val_pow(Value *v1, double exponent);
Value *val_exp(Value *v1);
Value *val_tanh(Value *v1);
void value_backward(Value *root);
void free_value_graph(Value *root);
void free_value_graphs(Value *const *roots, size_t n_roots);

int neuron_init(Neuron *neuron, size_t n_in);
Value *neuron_forward(const Neuron *neuron, Value *const *inputs,
                      size_t n_inputs);
void neuron_free(Neuron *neuron);

int layer_init(Layer *layer, size_t n_in, size_t n_out);
ValueArray layer_forward(const Layer *layer, Value *const *inputs,
                         size_t n_inputs);
void value_array_free(ValueArray *values);
void layer_free(Layer *layer);

int mlp_init(MLP *mlp, const size_t *sizes, size_t n_layers);
ValueArray mlp_forward(const MLP *mlp, Value *const *inputs,
                       size_t n_inputs);
void mlp_free(MLP *mlp);

#endif
