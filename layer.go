package main

import (
	"fmt"
)

type DenseOption func(*DenseDef)

type Layer struct {
	Neurons      []*Neuron
	ErrsFromNext [][]chan float64
	InsToNext    [][]chan float64
}

type DenseDef struct {
	InputNeurons int
	Neurons      int
	Activation   ActivationFunc
	Gradient     ActivationFunc
	HasInputSpec bool
}

func Activation(name string) DenseOption {
	act, ok := activationMap[name]
	if !ok {
		panic(fmt.Sprintf("Missing activation function for constant: %s", name))
	}

	return func(d *DenseDef) {
		d.Activation = act.fn
		d.Gradient = act.df
	}
}

func Dense(neurons int, options ...DenseOption) DenseDef {
	d := DenseDef{
		Neurons:      neurons,
		Activation:   linear,   // Default activation
		Gradient:     dfLinear, // Default derivative
		HasInputSpec: false,
	}

	// Apply all functional options
	for _, opt := range options {
		opt(&d)
	}

	return d
}

func InputDim(dim int) DenseOption {
	return func(d *DenseDef) {
		d.InputNeurons = dim
		d.HasInputSpec = true
	}
}

func NewInputLayer(m, n int) *Layer {
	inToNext := make([][]chan float64, m)
	for i := range m {
		inToNext[i] = make([]chan float64, n)
		for j := range n {
			inToNext[i][j] = make(chan float64, channelCapacity) // buffered
		}
	}

	errsFromNext := make([][]chan float64, m)
	for i := range m {
		errsFromNext[i] = make([]chan float64, n)
		for j := range n {
			// Initialize channels to receive the final error signal from the first hidden layer
			errsFromNext[i][j] = make(chan float64, channelCapacity)
		}
	}
	return &Layer{
		InsToNext:    inToNext,
		ErrsFromNext: errsFromNext,
	}
}

func NewLayer(errsToPrev, outsFromPrev [][]chan float64, outputNeurons int, f, df ActivationFunc) *Layer {
	numNeurons := len(outsFromPrev)

	layer := &Layer{
		Neurons:      make([]*Neuron, numNeurons),
		ErrsFromNext: make([][]chan float64, numNeurons),
		InsToNext:    make([][]chan float64, numNeurons),
	}

	// 1. Initialize each Neuron in the layer
	for j := 0; j < numNeurons; j++ {
		errsFromNext := make([]chan float64, outputNeurons)
		insToNext := make([]chan float64, outputNeurons)

		for i := range outputNeurons {
			errsFromNext[i] = make(chan float64, channelCapacity)
			insToNext[i] = make(chan float64, channelCapacity)
		}
		layer.Neurons[j] = NewNeuron(errsToPrev[j], outsFromPrev[j], errsFromNext, insToNext, f, df)
		layer.ErrsFromNext[j] = errsFromNext
		layer.InsToNext[j] = insToNext
	}

	// 2. Transpose to re-group channels by the next layer's neuron index
	layer.ErrsFromNext = Transpose(layer.ErrsFromNext)
	layer.InsToNext = Transpose(layer.InsToNext)

	return layer
}