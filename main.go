package main

import (
	"log"
	"math/rand"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	X, Y, err := LoadCSV("data.csv")
	if err != nil {
		log.Println("Error:", err)
		return
	}

	inputDim := len(X[0])
	numSamples := len(X[1:])

	log.Printf("Loaded dataset: %d samples, %d input features\n", numSamples, inputDim)

	nw := NewNetwork(
		Dense(16, InputDim(inputDim), Activation("relu")),
		Dense(8, Activation("relu")),
		Dense(4, Activation("relu")),
		Dense(2),
	)

	// Try to load previous weights if file exists
	if err := nw.LoadWeights("weights.json"); err != nil {
		log.Println("Error loading weights:", err)
	}

	// Setup Ctrl+C handler
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-c
		log.Println("\nSaving weights before exit...")
		if err := nw.SaveWeights("weights.json"); err != nil {
			log.Println("Error saving weights:", err)
		} else {
			log.Println("Weights saved successfully.")
		}
		os.Exit(0)
	}()

	cfg := TrainingConfig{
		Epochs:       100,
		BatchSize:    1,
		LearningRate: 0.001,
		LossFunction: CATEGORICAL_CROSS_ENTROPY,
		KClasses:     2, // For CATEGORICAL_CROSS_ENTROPY (Softmax Output).
		VerboseEvery: 10,
	}

	nw.Train(X, Y, cfg)

	loss, acc := nw.Evaluate(X, Y, cfg)
	log.Printf("Final Evaluation: Loss=%.6f, Accuracy=%.2f%%", loss, acc)

	// Test prediction
	// Pick random test data from training set
	rand := rand.Intn(len(X))
	test := X[rand]

	pred := nw.Predict(test, cfg)
	log.Printf("Test Input: %v | Predicted Output: %.4f | Actual Output: %.4f\n", test, pred, Y[rand])

	if err := nw.SaveWeights("weights.json"); err != nil {
		log.Println("Error saving weights:", err)
	} else {
		log.Println("Weights saved successfully.")
	}
}
