#include "micrograd.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "topological_sort/topo.h"

Value *init_value(double data_, double grad_, char op_, const char *label_,
                 Value **children_)
{
	Value *result = calloc(1, sizeof(*result));
	if (result == NULL) {
		return NULL;
	}

	result->data = data_;
	result->grad = grad_;
	result->op = op_;
	result->label = label_;

	if (children_ != NULL) {
		result->children[0] = children_[0];
		result->children[1] = children_[1];
	}

	return result;
}

void print_value(const value *val)
{
	if (val == NULL) {
		return;
	}

	printf("Value(data=%g, grad=%g)\n", val->data, val->grad);
}

Value *val_add(Value *v1, Value *v2)
{
	Value *children[] = {v1, v2};
	return init_value(v1->data + v2->data, 0.0, '+', NULL, children);
}

Value *val_mul(Value *v1, Value *v2)
{
	Value *children[] = {v1, v2};
	return init_value(v1->data * v2->data, 0.0, '*', NULL, children);
}

Value *val_pow(Value *v1, double exponent)
{
	Value *children[] = {v1, NULL};
	Value *result = init_value(pow(v1->data, exponent), 0.0, '^', NULL,
	                           children);
	if (result != NULL) {
		result->exponent = exponent;
	}
	return result;
}

Value *val_exp(Value *v1)
{
	Value *children[] = {v1, NULL};
	return init_value(exp(v1->data), 0.0, 'e', NULL, children);
}

Value *val_tanh(Value *v1)
{
	Value *children[] = {v1, NULL};
	return init_value(tanh(v1->data), 0.0, 't', NULL, children);
}

static void backward_node(Value *node)
{
	Value *left = node->children[0];
	Value *right = node->children[1];

	switch (node->op) {
	case '+':
		left->grad += node->grad;
		right->grad += node->grad;
		break;
	case '*':
		left->grad += right->data * node->grad;
		right->grad += left->data * node->grad;
		break;
	case '^':
		left->grad += node->exponent *
		              pow(left->data, node->exponent - 1.0) * node->grad;
		break;
	case 'e':
		left->grad += node->data * node->grad;
		break;
	case 't':
		left->grad += (1.0 - node->data * node->data) * node->grad;
		break;
	default:
		break;
	}
}

void value_backward(Value *root)
{
	TopologicalOrder order;

	if (root == NULL || !build_topological_order(root, &order)) {
		return;
	}

	root->grad = 1.0;
	for (size_t i = order.size; i > 0; i--) {
		backward_node(order.nodes[i - 1]);
	}

	free_topological_order(&order);
}

static int append_unique_value(Value ***values, size_t *size, Value *value)
{
	Value **new_values;

	for (size_t i = 0; i < *size; i++) {
		if ((*values)[i] == value) {
			return 1;
		}
	}

	new_values = realloc(*values, (*size + 1) * sizeof(**values));
	if (new_values == NULL) {
		return 0;
	}
	*values = new_values;
	(*values)[(*size)++] = value;
	return 1;
}

void free_value_graphs(Value *const *roots, size_t n_roots)
{
	Value **nodes = NULL;
	size_t n_nodes = 0;

	if (roots == NULL) {
		return;
	}

	for (size_t i = 0; i < n_roots; i++) {
		TopologicalOrder order;
		if (roots[i] == NULL || !build_topological_order(roots[i], &order)) {
			continue;
		}

		for (size_t j = 0; j < order.size; j++) {
			if (!append_unique_value(&nodes, &n_nodes, order.nodes[j])) {
				free_topological_order(&order);
				free(nodes);
				return;
			}
		}
		free_topological_order(&order);
	}

	for (size_t i = 0; i < n_nodes; i++) {
		free(nodes[i]);
	}
	free(nodes);
}

void free_value_graph(Value *root)
{
	free_value_graphs(&root, root == NULL ? 0 : 1);
}

static double random_weight(void)
{
	return 2.0 * (double)rand() / (double)RAND_MAX - 1.0;
}

static int append_parameter(ValueArray *params, Value *parameter)
{
	Value **new_values = realloc(params->values,
	                             (params->size + 1) * sizeof(*new_values));
	if (new_values == NULL) {
		return 0;
	}

	params->values = new_values;
	params->values[params->size++] = parameter;
	return 1;
}

static void clear_parameter_array(ValueArray *params)
{
	free(params->values);
	params->values = NULL;
	params->size = 0;
}

int neuron_init(Neuron *neuron, size_t n_in)
{
	if (neuron == NULL || n_in == 0) {
		return 0;
	}
	neuron->params = (ValueArray){NULL, 0};

	neuron->weights = calloc(n_in, sizeof(*neuron->weights));
	if (neuron->weights == NULL) {
		return 0;
	}

	neuron->n_in = n_in;
	neuron->bias = init_value(0.0, 0.0, '\0', "bias", NULL);
	if (neuron->bias == NULL) {
		free(neuron->weights);
		neuron->weights = NULL;
		return 0;
	}

	for (size_t i = 0; i < n_in; i++) {
		neuron->weights[i] = init_value(random_weight(), 0.0, '\0',
		                                "weight", NULL);
		if (neuron->weights[i] == NULL) {
			for (size_t j = 0; j < i; j++) {
				free(neuron->weights[j]);
			}
			free(neuron->bias);
			free(neuron->weights);
			neuron->weights = NULL;
			neuron->bias = NULL;
			return 0;
		}
	}

	for (size_t i = 0; i < n_in; i++) {
		if (!append_parameter(&neuron->params, neuron->weights[i])) {
			clear_parameter_array(&neuron->params);
			return 0;
		}
	}
	if (!append_parameter(&neuron->params, neuron->bias)) {
		clear_parameter_array(&neuron->params);
		return 0;
	}

	return 1;
}

Value *neuron_forward(const Neuron *neuron, Value *const *inputs,
	                  size_t n_inputs)
{
	Value *activation;

	if (neuron == NULL || inputs == NULL || n_inputs != neuron->n_in) {
		return NULL;
	}

	activation = neuron->bias;
	for (size_t i = 0; i < neuron->n_in; i++) {
		Value *product = val_mul(neuron->weights[i], inputs[i]);
		Value *next;

		if (product == NULL) {
			return NULL;
		}
		next = val_add(activation, product);
		if (next == NULL) {
			return NULL;
		}
		activation = next;
	}

	return val_tanh(activation);
}

void neuron_free(Neuron *neuron)
{
	if (neuron == NULL) {
		return;
	}

	free(neuron->weights);
	neuron->weights = NULL;
	clear_parameter_array(&neuron->params);
	neuron->bias = NULL;
	neuron->n_in = 0;
}

int layer_init(Layer *layer, size_t n_in, size_t n_out)
{
	if (layer == NULL || n_in == 0 || n_out == 0) {
		return 0;
	}

	layer->neurons = calloc(n_out, sizeof(*layer->neurons));
	if (layer->neurons == NULL) {
		return 0;
	}

	layer->n_in = n_in;
	layer->n_out = n_out;
	layer->params = (ValueArray){NULL, 0};
	for (size_t i = 0; i < n_out; i++) {
		if (!neuron_init(&layer->neurons[i], n_in)) {
			for (size_t j = 0; j < i; j++) {
				neuron_free(&layer->neurons[j]);
			}
			free(layer->neurons);
			layer->neurons = NULL;
			return 0;
		}
		for (size_t j = 0; j < layer->neurons[i].params.size; j++) {
			if (!append_parameter(&layer->params,
			                      layer->neurons[i].params.values[j])) {
				layer_free(layer);
				return 0;
			}
		}
	}

	return 1;
}

ValueArray layer_forward(const Layer *layer, Value *const *inputs,
	                     size_t n_inputs)
{
	ValueArray outputs = {NULL, 0};

	if (layer == NULL || inputs == NULL || n_inputs != layer->n_in) {
		return outputs;
	}

	outputs.values = calloc(layer->n_out, sizeof(*outputs.values));
	if (outputs.values == NULL) {
		return outputs;
	}

	outputs.size = layer->n_out;
	for (size_t i = 0; i < layer->n_out; i++) {
		outputs.values[i] = neuron_forward(&layer->neurons[i], inputs,
		                                   n_inputs);
		if (outputs.values[i] == NULL) {
			value_array_free(&outputs);
			return (ValueArray){NULL, 0};
		}
	}

	return outputs;
}

void value_array_free(ValueArray *values)
{
	if (values == NULL) {
		return;
	}

	free(values->values);
	values->values = NULL;
	values->size = 0;
}

void layer_free(Layer *layer)
{
	if (layer == NULL) {
		return;
	}

	for (size_t i = 0; i < layer->n_out; i++) {
		neuron_free(&layer->neurons[i]);
	}
	free(layer->neurons);
	layer->neurons = NULL;
	clear_parameter_array(&layer->params);
	layer->n_in = 0;
	layer->n_out = 0;
}

int mlp_init(MLP *mlp, const size_t *sizes, size_t n_layers)
{
	if (mlp == NULL || sizes == NULL || n_layers == 0) {
		return 0;
	}

	mlp->layers = calloc(n_layers, sizeof(*mlp->layers));
	if (mlp->layers == NULL) {
		return 0;
	}
	mlp->n_layers = n_layers;
	mlp->params = (ValueArray){NULL, 0};

	for (size_t i = 0; i < n_layers; i++) {
		if (!layer_init(&mlp->layers[i], sizes[i], sizes[i + 1])) {
			for (size_t j = 0; j < i; j++) {
				layer_free(&mlp->layers[j]);
			}
			free(mlp->layers);
			mlp->layers = NULL;
			mlp->n_layers = 0;
			return 0;
		}
		for (size_t j = 0; j < mlp->layers[i].params.size; j++) {
			if (!append_parameter(&mlp->params,
			                      mlp->layers[i].params.values[j])) {
				mlp_free(mlp);
				return 0;
			}
		}
	}

	return 1;
}

ValueArray mlp_forward(const MLP *mlp, Value *const *inputs,
	                   size_t n_inputs)
{
	ValueArray current = {(Value **)inputs, n_inputs};
	ValueArray next;

	if (mlp == NULL || inputs == NULL || mlp->n_layers == 0) {
		return (ValueArray){NULL, 0};
	}

	for (size_t i = 0; i < mlp->n_layers; i++) {
		next = layer_forward(&mlp->layers[i], current.values, current.size);
		if (i > 0) {
			value_array_free(&current);
		}
		if (next.values == NULL) {
			return (ValueArray){NULL, 0};
		}
		current = next;
	}

	return current;
}

void mlp_free(MLP *mlp)
{
	if (mlp == NULL) {
		return;
	}

	for (size_t i = 0; i < mlp->n_layers; i++) {
		layer_free(&mlp->layers[i]);
	}
	free(mlp->layers);
	mlp->layers = NULL;
	clear_parameter_array(&mlp->params);
	mlp->n_layers = 0;
}
