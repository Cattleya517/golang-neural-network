package nn

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)


func crossEntropyLoss(pred *mat.Dense, truth *mat.Dense) (float64, error) {
	// pred and truth should both be one-hot encoded vector
	r, _ := pred.Dims()
	losssum := 0.0
	for i := 0; i < r; i++ {
		predValue := pred.At(i, 0) 
		truthValue := truth.At(i, 0)
		// 防止 log(0) 導致 -Inf
		predValue = math.Max(predValue, 1e-15)
		losssum += -truthValue * math.Log(predValue)
	}
	return losssum, nil
}

func (nn *NeuralNetwork) forwardWithCache(input *mat.Dense) (*mat.Dense, []*mat.Dense, error) {
	// 用於訓練的forward版本 會記錄每層輸出以供backpropagation使用
	layerOutputs := []*mat.Dense{input}  // 保存輸入層
	current := input

	// 處理所有隱藏層
	for _, layer := range nn.Hidden {
		weighted := mat.Dense{}
		weighted.Mul(layer.weight, current)
		weighted.Add(&weighted, layer.bias)

		activated := relu(&weighted)
		layerOutputs = append(layerOutputs, activated)
		current = activated
	}
	// 處理輸出層
	output := mat.Dense{}
	output.Mul(nn.OutputWeight, current)
	output.Add(&output, nn.OutputBias)

	return &output, layerOutputs, nil
}


func reluDerivative(m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	new_matrix := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			value := m.At(i, j)
			if value <= 0 {
				// x <= 0,  relu(x) = 0 導數等於 0
				new_matrix.Set(i, j, 0)
			} else {
				// x > 0, relu(x) = x 導數等於 1
				new_matrix.Set(i, j, 1)
			}
		}
	}
	return new_matrix
}

func hadamardProduct(matA, matB *mat.Dense) (*mat.Dense, error){
	rA, cA := matA.Dims()
	rB, cB := matB.Dims()
	if rA != rB || cA != cB {
		return nil, fmt.Errorf("Matrices of different shape can't perform hadamardProduct")
	}
	result := mat.NewDense(rA, cA, nil)
	for i := 0; i < rA; i++{
		for j:=0; j < cA; j++{
			result.Set(i, j, matA.At(i, j) * matB.At(i, j)) 
		}
	}
	return  result, nil
}

func (nn *NeuralNetwork) backPropagation(pred *mat.Dense, traget *mat.Dense, layerOutputs []*mat.Dense){
	// IMPORTANT!!!: Compute ALL gradients first using ORIGINAL weights, then apply updates

	layercount := len(nn.Hidden)

	// Storage for gradients (will apply all at once at the end)
	hiddenWeightGrads := make([]*mat.Dense, layercount)
	hiddenBiasGrads := make([]*mat.Dense, layercount)

	// outputError = pred - target (for softmax + cross-entropy)
	var outputError mat.Dense
	outputError.Sub(pred, traget)

	// Compute output layer gradients
	var outputWeightGrad mat.Dense
	outputWeightGrad.Mul(&outputError, layerOutputs[len(layerOutputs)-1].T())

	// Save a copy of original output weights for gradient computation
	origOutputWeight := mat.DenseCopyOf(nn.OutputWeight)

	// Compute hidden layer gradients (backwards, using original weights)
	var prevLayerError *mat.Dense

	for i := layercount-1; i >= 0; i-- {
		var layerError mat.Dense

		if i == layercount-1 {
			// Use original output weights
			layerError.Mul(origOutputWeight.T(), &outputError)
		} else {
			// Use original weights of layer i+1 (not yet updated because we haven't applied updates)
			layerError.Mul(nn.Hidden[i+1].weight.T(), prevLayerError)
		}

		reluDeriv := reluDerivative(layerOutputs[i+1])
		layerErrorPtr, _ := hadamardProduct(&layerError, reluDeriv)
		prevLayerError = layerErrorPtr

		// Compute weight gradient for this layer
		var layerGrad mat.Dense
		layerGrad.Mul(layerErrorPtr, layerOutputs[i].T())
		hiddenWeightGrads[i] = mat.DenseCopyOf(&layerGrad)
		hiddenBiasGrads[i] = mat.DenseCopyOf(layerErrorPtr)
	}

	// Now apply all updates at once
	// Update output layer
	var scaledOutputWeightGrad mat.Dense
	scaledOutputWeightGrad.Scale(nn.LearningRate, &outputWeightGrad)
	r, c := nn.OutputWeight.Dims()
	newOutputWeight := mat.NewDense(r, c, nil)
	newOutputWeight.Sub(nn.OutputWeight, &scaledOutputWeightGrad)
	nn.OutputWeight = newOutputWeight

	var scaledOutputBiasGrad mat.Dense
	scaledOutputBiasGrad.Scale(nn.LearningRate, &outputError)
	rb, cb := nn.OutputBias.Dims()
	newOutputBias := mat.NewDense(rb, cb, nil)
	newOutputBias.Sub(nn.OutputBias, &scaledOutputBiasGrad)
	nn.OutputBias = newOutputBias

	// Update hidden layers
	for i := 0; i < layercount; i++ {
		var scaledWeightGrad mat.Dense
		scaledWeightGrad.Scale(nn.LearningRate, hiddenWeightGrads[i])

		wr, wc := nn.Hidden[i].weight.Dims()
		updatedWeight := mat.NewDense(wr, wc, nil)
		updatedWeight.Sub(nn.Hidden[i].weight, &scaledWeightGrad)
		nn.Hidden[i].weight = updatedWeight

		var scaledBiasGrad mat.Dense
		scaledBiasGrad.Scale(nn.LearningRate, hiddenBiasGrads[i])

		br, bc := nn.Hidden[i].bias.Dims()
		updatedBias := mat.NewDense(br, bc, nil)
		updatedBias.Sub(nn.Hidden[i].bias, &scaledBiasGrad)
		nn.Hidden[i].bias = updatedBias
	}
}   


func (nn *NeuralNetwork) train(input *mat.Dense, target *mat.Dense) (float64, error) {
	//foward
	logits, layerOutputs, _ := nn.forwardWithCache(input)

	// softmax to decode preditction from pred matrix
	pred := softmax(logits)

	//record loss
	loss, _ := crossEntropyLoss(pred, target)

	//backpropagation
	nn.backPropagation(pred, target, layerOutputs)

	return loss, nil
}

type TrainingData struct{
	Input *mat.Dense
	Target *mat.Dense
}

func LoadingDataFromCSV(filename string) ([]TrainingData, error){
	file, err := os.Open(filename)
	if err != nil{
		return nil, fmt.Errorf("Can't open file %s", filename)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	var data []TrainingData
	for {
		record, err := reader.Read()
		if err == io.EOF{
			break
		}
		if err != nil{
			return nil, fmt.Errorf("Error reading csv %v", err)
		}
		var td TrainingData
		target := mat.NewDense(10, 1, nil)
		label, _ := strconv.Atoi(record[0])
		target.Set(label, 0, 1.0)
		td.Target = target

		input := mat.NewDense(784, 1, nil)
		for i:=0; i<784; i++{
			pixel, _ := strconv.Atoi(record[1:][i])
			input.Set(i, 0, float64(pixel)/255.0) 
		}
		td.Input = input
		data = append(data, td)
	}
	return data, nil
}

func TrainingLoop(nn *NeuralNetwork, epoch int, trainingset []TrainingData, testset[]TrainingData) error {
	// training loop

	for i := 0; i < epoch; i++ {
		//遍例所有training sample
		lossSum := 0.0
		for _, sample := range trainingset {
			singleLoss, err := nn.train(sample.Input, sample.Target)
			if err != nil {
				return fmt.Errorf("Error During Training")
			}
			lossSum += singleLoss
		}
		avgLoss := lossSum / float64(len(trainingset))
		fmt.Printf("Epoch 【%d/%d】| Average training Loss on this epoch %.4f\n", i+1, epoch, avgLoss)
	
		// validation loop
		acc, err := validate(nn, testset)
		if err != nil {
			return fmt.Errorf("Error During Validation")
		}
		fmt.Printf("Validation 【%d/%d】 | Accuracy on validation set on this epoch: %.2f\n", i+1, epoch, acc)
	}

	return nil
}

func argmax(input *mat.Dense) (int, error) {
	r, c := input.Dims()
	if r!=10 || c != 1 {
		return 0, fmt.Errorf("Not a row vector, not avaliable in mnist task.")
	}

	maxValue := input.At(0, 0)
	maxIdx := 0
	for i:=0; i<r; i++{
		value := input.At(i, 0)
		if value > maxValue{
			maxValue = value
			maxIdx = i
		}
	}
	return maxIdx, nil
}


func validate(nn *NeuralNetwork, testset []TrainingData) (float64, error) {

	correct := 0
	for _, sample := range testset {
		logit, err := nn.forward(sample.Input)
		if err != nil{
			return 0, fmt.Errorf("Error during Inference")
		}
		pred, err := argmax(softmax(logit))
		if err != nil {
			return 0, err
		}

		ans, err := argmax(sample.Target)
		if err != nil {
			return 0, err
		}
		if pred == ans{
			correct++
		}	

	}
	accuracy := float64(correct) / float64(len(testset)) 
	return accuracy, nil
}

