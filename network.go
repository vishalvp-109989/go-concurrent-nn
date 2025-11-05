package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"time"
)

// Loss function constants (enum)
const (
	MSE                       = iota // Mean Squared Error (Regression)
	CATEGORICAL_CROSS_ENTROPY        // Multi-Class Classification
	BINARY_CROSS_ENTROPY             // Binary Classification
)

type Network struct {
	InputLayer  *Layer
	Hidden      []*Layer
	OutputLayer *Layer
}

type TrainingConfig struct {
	Epochs       int
	BatchSize    int
	LearningRate float64
	LossFunction int
	KClasses     int
	VerboseEvery int
}

type SerializableNeuron struct {
	Weights []float64 `json:"weights"`
	Bias    float64   `json:"bias"`
}

type SerializableLayer struct {
	Neurons []SerializableNeuron `json:"neurons"`
}

type SerializableNetwork struct {
	Layers []SerializableLayer `json:"layers"`
}

// NewNetwork builds: InputLayer -> Layer(defs[1]) -> Layer(defs[2]) -> ... -> Layer(defs[len-1]) -> OutputLayer
func NewNetwork(defs ...DenseDef) *Network {
	if len(defs) == 1 && defs[0].Neurons != 1 {
		panic("Provide at least an input spec and one Dense layer for m!=1")
	}
	// First Dense must carry input spec (m,inputDim)
	if !defs[0].HasInputSpec {
		panic("First Dense must be called with two args: Dense(m, inputDim)")
	}

	// Create Input layer
	iLayer := NewInputLayer(defs[0].Neurons, defs[0].InputNeurons)

	prevErrs := iLayer.ErrsFromNext
	prevIns := iLayer.InsToNext

	var hidden []*Layer

	// For every Dense def except the first (input spec),
	// create a regular Layer (including the last Dense def).
	for i := range len(defs) - 1 {
		l := NewLayer(prevErrs, prevIns, defs[i+1].Neurons, defs[i+1].Activation, defs[i+1].Gradient)
		hidden = append(hidden, l)

		// advance the prevs to this layer's outputs for the next iteration
		prevErrs = l.ErrsFromNext
		prevIns = l.InsToNext
	}

	// Last layer.
	l := NewLayer(prevErrs, prevIns, 1, defs[len(defs)-1].Activation, defs[len(defs)-1].Gradient)
	hidden = append(hidden, l)

	return &Network{
		InputLayer:  iLayer,
		Hidden:      hidden,
		OutputLayer: hidden[len(hidden)-1],
	}
}

func (nw *Network) SaveWeights(filename string) error {
	sn := SerializableNetwork{}

	for _, layer := range nw.Hidden {
		sl := SerializableLayer{}
		for _, neuron := range layer.Neurons {
			sl.Neurons = append(sl.Neurons, SerializableNeuron{
				Weights: neuron.Weights,
				Bias:    neuron.Bias,
			})
		}
		sn.Layers = append(sn.Layers, sl)
	}

	data, err := json.MarshalIndent(sn, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filename, data, 0644)
}

func (nw *Network) LoadWeights(filename string) error {
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		log.Println("Weights file not found, starting fresh.")
		return nil
	}

	data, err := os.ReadFile(filename)
	if err != nil {
		return err
	}

	var sn SerializableNetwork
	if err := json.Unmarshal(data, &sn); err != nil {
		return err
	}

	// Copy values back into your network
	for i, sl := range sn.Layers {
		if i >= len(nw.Hidden) {
			return fmt.Errorf("mismatch: saved network has more layers (%d) than current network (%d)", len(sn.Layers), len(nw.Hidden))
		}
		for j, snNeuron := range sl.Neurons {
			if j >= len(nw.Hidden[i].Neurons) {
				return fmt.Errorf("mismatch: layer %d has more neurons in saved file (%d) than in current network (%d)",
					i, len(sl.Neurons), len(nw.Hidden[i].Neurons))
			}

			n := nw.Hidden[i].Neurons[j]

			if len(n.Weights) != len(snNeuron.Weights) {
				return fmt.Errorf("weight mismatch at layer %d neuron %d: expected %d weights, got %d",
					i, j, len(n.Weights), len(snNeuron.Weights))
			}

			// Safe to assign now
			copy(n.Weights, snNeuron.Weights)
			n.Bias = snNeuron.Bias
		}
	}

	log.Println("Weights loaded successfully.")
	return nil
}

func (nw *Network) GetOutput() []float64 {
	numNeurons := len(nw.OutputLayer.InsToNext[0])

	outputs := make([]float64, numNeurons)
	for j := range numNeurons {
		outputs[j] = <-nw.OutputLayer.InsToNext[0][j]
	}
	return outputs
}

func (nw *Network) FeedForward(x []float64) {
	iLayer := nw.InputLayer
	m := len(iLayer.InsToNext)

	if m == 0 || len(iLayer.InsToNext[0]) == 0 {
		return
	}
	n := len(iLayer.InsToNext[0])

	for k := range m {
		for j := range n {
			iLayer.InsToNext[k][j] <- x[j]
		}
	}
}

func (nw *Network) Feedback(pred []float64, target []float64) {
	oLayer := nw.OutputLayer
	numOutputNeurons := len(oLayer.InsToNext[0])

	// Calculate the error vector: error = pred - target
	finalErrors := make([]float64, numOutputNeurons)
	for i := range numOutputNeurons {
		finalErrors[i] = pred[i] - target[i]
	}

	// Inject the final error into the output neurons to start backpropagation
	for i := range numOutputNeurons {
		oLayer.Neurons[i].ErrsFromNext[0] <- finalErrors[i]
	}
}

func (nw *Network) WaitForBackpropFinish() {
	iLayer := nw.InputLayer

	// m is the number of rows/parallel inputs
	m := len(iLayer.ErrsFromNext)

	// n is the number of columns/features
	if m == 0 || len(iLayer.ErrsFromNext[0]) == 0 {
		panic("Should not happen in a valid network")
	}
	n := len(iLayer.ErrsFromNext[0])

	// Wait for the backpropagation signal on ALL m*n channels
	for k := range m {
		for j := range n {
			<-iLayer.ErrsFromNext[k][j]
		}
	}
}

func (nw *Network) Train(X [][]float64, Y []float64, cfg TrainingConfig) {
	nw.UpdateLR(cfg)
	batchSize = cfg.BatchSize

	start := time.Now()
	for epoch := range cfg.Epochs {
		totalLoss := 0.0
		correct := 0
		for i := range Y[1:] {
			x := X[i]
			target := Y[i]

			// 1. Forward pass
			nw.FeedForward(x)
			pred := nw.GetOutput()

			// 2. Compute loss
			loss, targetVector := nw.computeLoss(pred, target, cfg)
			totalLoss += loss

			// 3. Backward pass
			nw.Feedback(pred, targetVector)
			nw.WaitForBackpropFinish()

			// 4. Accuracy Check
			if nw.isCorrect(pred, target, cfg) {
				correct++
			}
		}
		// Logging
		if epoch%cfg.VerboseEvery == 0 {
			avgLoss := totalLoss / float64(len(Y[1:]))
			acc := float64(correct) / float64(len(Y[1:])) * 100.0
			elapsed := time.Since(start).Minutes()
			log.Printf("Epoch %d | Loss: %.6f | Accuracy: %.2f%% | Time: %.2f min\n", epoch, avgLoss, acc, elapsed)

			// if err := nw.SaveWeights("weights.json"); err != nil {
			// 	log.Println("Error saving weights:", err)
			// } else {
			// 	log.Println("Weights saved successfully.")
			// }
		}
	}
}

func (nw *Network) Predict(x []float64, cfg TrainingConfig) float64 {
	nw.FeedForward(x)
	predVector := nw.GetOutput()
	switch cfg.LossFunction {
	case CATEGORICAL_CROSS_ENTROPY:
		predVector = Softmax(predVector)
		predictedClass := OneHotDecode(predVector)

		return predictedClass

	case BINARY_CROSS_ENTROPY:
		predScalar := predVector[0] // Probability output

		// Determine the binary class based on the 0.5 threshold
		predictedClass := 0
		if predScalar >= 0.5 {
			predictedClass = 1
		}

		return float64(predictedClass)
	case MSE:
		predScalar := predVector[0]
		return predScalar

	default:
		// Handle uninitialized or unknown loss function
		log.Printf("Warning: Cannot log test result. Unknown loss function: %d\n", cfg.LossFunction)
	}
	return 0.0
}

func (nw *Network) Evaluate(X [][]float64, Y []float64, cfg TrainingConfig) (float64, float64) {
	totalLoss := 0.0
	correct := 0

	for i := range Y[1:] {
		x := X[i]
		target := Y[i]

		// 1. Forward pass
		nw.FeedForward(x)
		pred := nw.GetOutput()

		// 2. Compute loss
		loss, _ := nw.computeLoss(pred, target, cfg)
		totalLoss += loss

		// 3. Accuracy Check
		if nw.isCorrect(pred, target, cfg) {
			correct++
		}
	}
	avgLoss := totalLoss / float64(len(Y[1:]))
	acc := float64(correct) / float64(len(Y[1:])) * 100.0
	return avgLoss, acc
}

func (nw *Network) computeLoss(predVector []float64, targetScalar float64, cfg TrainingConfig) (float64, []float64) {
	switch cfg.LossFunction {
	case CATEGORICAL_CROSS_ENTROPY:
		// Ensure probabilities sum to 1
		predVector = Softmax(predVector)

		// Convert scalar target to OHE
		targetVector := OneHotEncode(targetScalar, cfg.KClasses)

		loss := 0.0
		// L = - Sum[ y_i * log(y_hat_i) ]
		for k := range cfg.KClasses {
			y := targetVector[k]
			y_hat := predVector[k]

			// Add small epsilon to prevent log(0)
			epsilon := 1e-15
			y_hat_safe := math.Max(epsilon, y_hat)

			loss += -y * math.Log(y_hat_safe)
		}
		return loss, targetVector

	case BINARY_CROSS_ENTROPY:
		predScalar := predVector[0] // y_hat

		// Add small epsilon to prevent log(0)
		epsilon := 1e-15

		// y * log(y_hat)
		term1 := targetScalar * math.Log(math.Max(epsilon, predScalar))

		// (1-y) * log(1 - y_hat)
		term2 := (1.0 - targetScalar) * math.Log(math.Max(epsilon, 1.0-predScalar))

		loss := -(term1 + term2)

		// Set targetVector
		return loss, []float64{targetScalar}

	case MSE:
		predScalar := predVector[0]
		return 0.5 * math.Pow(predScalar-targetScalar, 2), []float64{targetScalar}

	default:
		panic(fmt.Sprintf("Unknown loss function: %d", cfg.LossFunction))
	}
}

func (nw *Network) isCorrect(predVector []float64, targetScalar float64, cfg TrainingConfig) bool {
	switch cfg.LossFunction {
	case CATEGORICAL_CROSS_ENTROPY:
		predVector = Softmax(predVector)

		predictedClass := OneHotDecode(predVector)

		// Compare predicted class index to the actual class index
		return float64(predictedClass) == targetScalar

	case BINARY_CROSS_ENTROPY:
		predScalar := predVector[0] // The single probability output by the Sigmoid layer

		// The predicted class is 1 if the probability is >= 0.5, otherwise 0.
		predictedClass := 0
		if predScalar >= 0.5 {
			predictedClass = 1
		}

		// Compare predicted class (0 or 1) to the actual binary target (0 or 1)
		return float64(predictedClass) == targetScalar

	case MSE:
		// For regression, "accuracy" = % of predictions close to target
		predScalar := predVector[0]
		return math.Abs(predScalar-targetScalar) < 1
	}
	return false
}

func (nw *Network) UpdateLR(cfg TrainingConfig) {
	for _, layer := range nw.Hidden {
		for _, neuron := range layer.Neurons {
			neuron.LR = cfg.LearningRate
		}
	}
}
