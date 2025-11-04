package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"
)

// Loss function constants (enum)
const (
	MSE                       = iota // Mean Squared Error (Regression)
	CATEGORICAL_CROSS_ENTROPY        // Multi-Class Classification
	BINARY_CROSS_ENTROPY             // Binary Classification
)

const learning_rate = 0.001
const lossFunction = CATEGORICAL_CROSS_ENTROPY

// K_CLASSES = 1: BINARY_CROSS_ENTROPY (Sigmoid Output),  K_CLASSES >= 2: CATEGORICAL_CROSS_ENTROPY (Softmax Output).
// Ouput layer should have K_CLASSES neurons
const K_CLASSES = 2

const BATCH = 1
const EPOCH = 100

func main() {
	X, Y, err := LoadCSV("data.csv")
	if err != nil {
		log.Println("Error:", err)
		return
	}

	inputDim := len(X[0])
	numSamples := len(X[1:])

	log.Printf("Loaded dataset: %d samples, %d input features\n", numSamples, inputDim)

	// m := 1
	// n := inputDim

	network := NewNetwork(
		Dense(16, InputDim(inputDim)),
		Dense(8, Activation("relu")),
		Dense(4, Activation("relu")),
		Dense(2),
	)

	// Try to load previous weights if file exists
	if err := network.LoadWeights("weights.json"); err != nil {
		log.Println("Error loading weights:", err)
	}

	// Setup Ctrl+C handler
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-c
		log.Println("\nSaving weights before exit...")
		if err := network.SaveWeights("weights.json"); err != nil {
			log.Println("Error saving weights:", err)
		} else {
			log.Println("Weights saved successfully.")
		}
		os.Exit(0)
	}()

	start := time.Now()
	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for epoch := range EPOCH {
			totalLoss := 0.0
			correct := 0
			var predScalar, targetScalar float64
			var predVector, targetVector []float64
			for i := range Y[1:] {
				targetScalar = Y[i]

				x := X[i]
				network.FeedForward(x)

				predVector = network.GetOutput()

				switch lossFunction {
				case CATEGORICAL_CROSS_ENTROPY:
					// Ensure probabilities sum to 1
					predVector = Softmax(predVector)

					// Convert scalar target to OHE
					targetVector = make([]float64, K_CLASSES)
					if targetScalar >= 0 && targetScalar < float64(K_CLASSES) {
						targetVector[int(targetScalar)] = 1.0
					}

					loss := 0.0
					// L = - Sum[ y_i * log(y_hat_i) ]
					for k := range K_CLASSES {
						y := targetVector[k]
						y_hat := predVector[k]

						// Add small epsilon to prevent log(0)
						epsilon := 1e-15
						y_hat_safe := math.Max(epsilon, y_hat)

						loss += -y * math.Log(y_hat_safe)
					}
					totalLoss += loss

				case BINARY_CROSS_ENTROPY:
					predScalar = predVector[0]   // y_hat
					targetScalar := targetScalar // y

					// Add small epsilon to prevent log(0)
					epsilon := 1e-15

					// y * log(y_hat)
					term1 := targetScalar * math.Log(math.Max(epsilon, predScalar))

					// (1-y) * log(1 - y_hat)
					term2 := (1.0 - targetScalar) * math.Log(math.Max(epsilon, 1.0-predScalar))

					loss := -(term1 + term2)
					totalLoss += loss

					// Set targetVector for gradient calculation
					targetVector = []float64{targetScalar}

				case MSE:
					predScalar = predVector[0]
					targetVector = []float64{targetScalar}
					totalLoss += 0.5 * math.Pow(predScalar-targetScalar, 2)

				default:
					panic(fmt.Sprintf("Unknown loss function: %d", lossFunction))
				}

				// 2. BACKWARD PASS INJECTION (Error: y_hat - y)
				network.Feedback(predVector, targetVector)

				// 3. SYNCHRONIZATION BARRIER
				network.WaitForBackpropFinish()

				// Accuracy Check
				switch lossFunction {
				case CATEGORICAL_CROSS_ENTROPY:
					predVector = Softmax(predVector)

					predictedClass := 0
					maxProb := predVector[0]

					// Find the class index with the highest probability
					for k := 1; k < K_CLASSES; k++ {
						if predVector[k] > maxProb {
							maxProb = predVector[k]
							predictedClass = k
						}
					}

					// Compare predicted class index to the actual class index
					if float64(predictedClass) == targetScalar {
						correct++
					}

				case BINARY_CROSS_ENTROPY:
					predScalar := predVector[0] // The single probability output by the Sigmoid layer

					// The predicted class is 1 if the probability is >= 0.5, otherwise 0.
					predictedClass := 0
					if predScalar >= 0.5 {
						predictedClass = 1
					}

					// Compare predicted class (0 or 1) to the actual binary target (0 or 1)
					if float64(predictedClass) == targetScalar {
						correct++
					}

				case MSE:
					// For regression, "accuracy" = % of predictions close to target
					predScalar := predVector[0]
					if math.Abs(predScalar-targetScalar) < 1 {
						correct++
					}
				}
			}
			if epoch%10 == 0 {
				avgLoss := totalLoss / float64(len(Y[1:]))
				acc := float64(correct) / float64(len(Y[1:])) * 100.0
				elapsed := time.Since(start).Minutes()
				log.Printf("Epoch %d | Loss: %.6f | Accuracy: %.2f%% | Time: %.2f min\n", epoch, avgLoss, acc, elapsed)
			}
		}
	}()
	wg.Wait()
	// Test prediction
	// Pick random test data from training set
	rand := rand.Intn(len(X))
	test := X[rand]
	network.FeedForward(test)
	predVector := network.GetOutput()
	switch lossFunction {
	case CATEGORICAL_CROSS_ENTROPY:
		predVector = Softmax(predVector)
		predictedClass := 0
		maxProb := predVector[0]

		// Find the class index with the highest probability
		for k := 1; k < K_CLASSES; k++ {
			if predVector[k] > maxProb {
				maxProb = predVector[k]
				predictedClass = k
			}
		}

		log.Printf("Test Input: %v | Predicted Class: %v | Actual Output: %.4f\n", test, predictedClass, Y[rand])

	case BINARY_CROSS_ENTROPY:
		predScalar := predVector[0] // Probability output

		// Determine the binary class based on the 0.5 threshold
		predictedClass := 0
		if predScalar >= 0.5 {
			predictedClass = 1
		}

		log.Printf("Test Input: %v | Predicted Class: %v (Prob: %.4f) | Actual Output: %.4f\n", test, predictedClass, predScalar, Y[rand])

	case MSE:
		predScalar := predVector[0]
		log.Printf("Test Input: %v | Predicted Output: %.4f | Actual Output: %.4f\n", test, predScalar, Y[rand])

	default:
		// Handle uninitialized or unknown loss function
		log.Printf("Warning: Cannot log test result. Unknown loss function: %d\n", lossFunction)
	}

	if err := network.SaveWeights("weights.json"); err != nil {
		log.Println("Error saving weights:", err)
	} else {
		log.Println("Weights saved successfully.")
	}
}
