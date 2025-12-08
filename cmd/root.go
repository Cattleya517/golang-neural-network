package cmd

import (
	"errors"
	"fmt"
	"golang-neural-network/nn"
	"os"
	"path/filepath"
	"strconv"

	"github.com/AlecAivazis/survey/v2"
	"github.com/spf13/cobra"
)

// rootCmd 是 Cobra 的根命令
// 當用戶執行程序時，這個命令會被執行
var rootCmd = &cobra.Command{
	Use:   "mnist-nn",
	Short: "MNIST Hand Written Digit Recognition System",
	Run: func(cmd *cobra.Command, args []string) {
		// 調用主選單函數
		showMainMenu()
	},
}

// main.go 會調用這個函數來啟動 CLI
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// showMainMenu 顯示互動式主選單
func showMainMenu() {
	// Create choice to save user choice
	var choice string

	// Create a "select" question
	prompt := &survey.Select{
		Message: "Welcome to MNIST playgorund! Choose your Action",
		Options: []string{
			"1. Train self-defined model and test with your own hand written digit.",
			"2. Load trained model and test with your own hand written digit.",
			"3. Exit",
		},
	}

	// verify user choice validity
	err := survey.AskOne(prompt, &choice)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	// Outcome based on user choice
	switch choice {
	case "1. Train self-defined model and test with your own hand written digit.":
		fmt.Println("\nTraining mode selected")
		fmt.Println("Define your model structure")
		trainingFlow()
		
	case "2. Load trained model and test with your own hand written digit.":
		fmt.Println("\nLoading model and GUI")
		// TODO: 稍後會調用載入模型函數
		
	case "3. Exit":
		fmt.Println("\nBye！")
		return
		
	default:
		fmt.Println("Unknow Response, Aborting...")
		return
	}
}


func positiveIntValidator(val interface{}) error {
    str, ok := val.(string)
    if !ok || str == "" {
        return fmt.Errorf("Please enter number")
    }
    num, err := strconv.Atoi(str)
    if err != nil {
        return fmt.Errorf("Invalid integer")
    }
    if num <= 0 {
        return fmt.Errorf("Must be greater than 0")
    }
    return nil
}

func positiveFloatValidator(val interface{}) error {
    str, ok := val.(string)
    if !ok || str == "" {
        return fmt.Errorf("Please enter number")
    }
    num, err := strconv.ParseFloat(str, 64)
    if err != nil {
        return fmt.Errorf("Invalid float")
    }
    if num <= 0 {
        return fmt.Errorf("Must be greater than 0")
    }
    return nil
}

func trainingFlow(){
	//接收參數：幾層hidden, [hiddenint], learningRate, epoch
	var hiddenLayerNumStr string
	var hiddenLayerNum int
	survey.AskOne(&survey.Input{
        Message: "Enter Hidden Layer Number:",
        Default: "2",
    }, &hiddenLayerNumStr, survey.WithValidator(positiveIntValidator))
    
    hiddenLayerNum, _ = strconv.Atoi(hiddenLayerNumStr) 

	// Collect node count for each hidden layer
	var layerNodesAmount []int
	for i := 0; i < hiddenLayerNum; i++ {
		var nodeCountStr string
		survey.AskOne(&survey.Input{
			Message: fmt.Sprintf("Enter node count for hidden layer %d:", i+1),
			Default: "8",
		}, &nodeCountStr, survey.WithValidator(positiveIntValidator))
		
		nodeCount, _ := strconv.Atoi(nodeCountStr)
		layerNodesAmount = append(layerNodesAmount, nodeCount)
	}

	var learningRateStr string
	var learningRate float64
	survey.AskOne(&survey.Input{
		Message: "Enter Learning Rate:",
		Default: "0.01",
	}, &learningRateStr, survey.WithValidator(positiveFloatValidator))

	learningRate, _ = strconv.ParseFloat(learningRateStr, 64) 

	var epochStr string
	var epoch int
	survey.AskOne(&survey.Input{
        Message: "Enter Epoch:",
        Default: "5",
    }, &epochStr, survey.WithValidator(positiveIntValidator))
    
    epoch, _ = strconv.Atoi(epochStr)

	fmt.Printf("\nTraining Configuration:\n")
	fmt.Printf("Hidden Layers: %d\n", hiddenLayerNum)
	fmt.Printf("Layer Nodes: %v\n", layerNodesAmount)
	fmt.Printf("Learning Rate: %f\n", learningRate)
	fmt.Printf("Epochs: %d\n", epoch)
	
	// Create network
	network, err := nn.NewNeuralNetwork(784, 10, layerNodesAmount, learningRate)
	if err != nil {
		fmt.Printf("Error creating neural network: %v\n", err)
		return
	}
	
	fmt.Println("\nNeural network created successfully!")
	
	// Load training set
	fmt.Println("\nLoading training data...")

	// check if the user dont have train and test data
	trainFilePath := "mnist_data/train.csv"
	testFilePath := "mnist_data/test.csv"
	
	if _, err := os.Stat(trainFilePath); errors.Is(err, os.ErrNotExist) {
		fmt.Println("train CSV file not found, parsing MNIST images and labels")
		err = nn.ConvertToCSV("mnist_data/train-images.idx3-ubyte", "mnist_data/train-labels.idx1-ubyte", "mnist_data/train.csv")
		if err != nil {
			fmt.Printf("Error converting training data: %v\n", err)
			return
		}
	} 
	if _, err := os.Stat(testFilePath); errors.Is(err, os.ErrNotExist) {
		fmt.Println("test CSV file not found, parsing MNIST images and labels")
		err = nn.ConvertToCSV("mnist_data/t10k-images.idx3-ubyte", "mnist_data/t10k-labels.idx1-ubyte", "mnist_data/test.csv")
		if err != nil {
			fmt.Printf("Error converting test data: %v\n", err)
			return
		}
	} 

	// loading training set
	trainingSet, err := nn.LoadingDataFromCSV("mnist_data/train.csv")
	if err != nil {
		fmt.Printf("Error loading training data: %v\n", err)
		return
	}
	fmt.Printf("Loaded %d training samples\n", len(trainingSet))
	
	// load validation set
	fmt.Println("Loading test data...")
	valSet, err := nn.LoadingDataFromCSV("mnist_data/test.csv")
	if err != nil {
		fmt.Printf("Error loading test data: %v\n", err)
		return
	}
	fmt.Printf("Loaded %d test samples\n", len(valSet))
	
	fmt.Println("\nStarting training...")
	err = nn.TrainingLoop(network, epoch, trainingSet, valSet)
	if err != nil {
		fmt.Printf("Error during training: %v\n", err)
		return
	}
	fmt.Println("\nTraining completed!")
	


	// 儲存模型
	var modelName string
	survey.AskOne(&survey.Input{
        Message: "Enter model filename (will be saved in models/):",
        Default: "my_mnist_model.json",
    }, &modelName)
	
	modelName += ".json"
	modelsDir := "models"
	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		fmt.Printf("Error creating models directory: %v\n", err)
		return
	}
	modelPath := filepath.Join(modelsDir, modelName)
	
	fmt.Printf("\nSaving model to %s...\n", modelPath)
	err = nn.SaveModel(network, modelPath)
	if err != nil {
		fmt.Printf("Error saving model: %v\n", err)
		return
	}
	
	fmt.Printf("Model saved successfully to %s\n", modelPath)
}