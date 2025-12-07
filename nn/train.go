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
		layerOutputs = append(layerOutputs, activated)  // 保存這層的輸出
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
		return mat.Dense{}, fmt.Errorf("Matrices of different shape can't perform hadamardProduct")
	}
	result := mat.NewDense(rA, cA, nil)
	for i := 0; i < r; i++{
		for j:=0; j < c; j++{
			result.Set(i, j, matA.At(i, j) * matB.At(i, j)) 
		}
	}
	return  result, nil
}

func (nn *NeuralNetwork) backPropagation(pred *mat.Dense, traget *mat.Dense, layerOutputs []*mat.Dense){

	// outputError
	var outputError mat.Dense
	outputError.Sub(pred, traget)  // This calculation only works for Cross-Entropy loss function

	//outputGrad = outputError * last hidden layer output ^ T
	var outputWeightGrad mat.Dense
	outputWeightGrad.Mul(&outputError, layerOutputs[len(layerOutputs)-1].T())

	var scaledWeightGrad mat.Dense
	scaledWeightGrad.Scale(nn.LearningRate, &outputWeightGrad)
	nn.OutputWeight.Sub(nn.OutputWeight, &scaledWeightGrad)
	
	var scaledBiasGrad mat.Dense
	scaledBiasGrad.Scale(nn.LearningRate, &outputError)
	nn.OutputBias.Sub(nn.OutputBias, &scaledBiasGrad)

	// 從最後一層hidden layer 逐步修正到第一層
	
	var prevLayerError *mat.Dense
	layercount := len(nn.Hidden)

	for i := layercount-1 ; i > 0 ; i-=1 {
		// 如果是最後一層  需要先從ouput layer 取得gradient訊息

		if i == layercount-1 {
			var layerError *mat.Dense 
			layerError.Mul(nn.OutputWeight.T(), &outputError)
			
			reluDeriv := reluDerivative(layerOutputs[i])
			layerError, _ = hadamardProduct(&layerError, reluDeriv)
			prevLayerError = layerError

			var layerGrad mat.Dense
			layerGrad.Mul(layerError, layerOutputs[i-1].T())

			var scaledWeightGrad mat.Dense
			scaledWeightGrad.Scale(nn.LearningRate, &layerGrad)
			nn.Hidden[i].weight.Sub(nn.Hidden[i].weight, &scaledWeightGrad)

			var scaledBiasGrad mat.Dense
			scaledBiasGrad.Scale(nn.LearningRate, &layerError)
			nn.Hidden[i].bias.Sub(nn.Hidden[i].bias, &scaledBiasGrad)

		}	else {
			// layerError = 後一層weight.T() * 後一層的error
			// layerGrad (on L) = layerError (on L) * layerOutput.T() (on L-1)
		 
			var layerError *mat.Dense
			layerError.Mul(nn.Hidden[i+1].weight.T(), prevLayerError)
			reluDeriv := reluDerivative(layerOutputs[i])
			layerError, _ = hadamardProduct(&layerError, reluDeriv)
			prevLayerError = layerError

			var layerGrad mat.Dense
			layerGrad.Mul(layerError, layerOutputs[i-1].T())

			var scaledWeightGrad mat.Dense
			scaledWeightGrad.Scale(nn.LearningRate, &layerGrad)
			nn.Hidden[i].weight.Sub(nn.Hidden[i].weight, &scaledWeightGrad)

			var scaledBiasGrad mat.Dense
			scaledBiasGrad.Scale(nn.LearningRate, &layerError)
			nn.Hidden[i].bias.Sub(nn.Hidden[i].bias, &scaledBiasGrad)

		} 
	}

}   


func (nn *NeuralNetwork) train(input *mat.Dense, target *mat.Dense) error {

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

func trainingLoop(nn NeuralNetwork, epoch int, trainingset []TrainingData) error {
	// training loop
	for i:=0; i<epoch; i++ {

		//遍例所有training sample
		
		lossSum := 0.0
		// for every sample in training set
		singleLoss, nil := nn.train()

		lossSum += singleLoss
		fmt.Printf("Epoch 【%d/%d】| Training Loss on this epoch %.4f\n", i, epoch, loss)
	}


}
