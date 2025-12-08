package nn

import (
	"encoding/json"
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"
)

type SerializableModel struct {
    Inputs       int           `json:"inputs"`
    OutputClass  int           `json:"output_class"`
    HiddenLayers []SerializableLayer `json:"hidden_layers"`
    OutputWeight [][]float64   `json:"output_weight"`
    OutputBias   [][]float64   `json:"output_bias"`
    LearningRate float64       `json:"learning_rate"`
}

type SerializableLayer struct {
    Weight [][]float64 `json:"weight"`
    Bias   [][]float64 `json:"bias"`
}


func denseToSlice(m *mat.Dense) [][]float64 {
	r, c := m.Dims()
	
	result := make([][]float64, r)
	
	for i := 0; i < r; i++ {
		result[i] = make([]float64, c)
		
		for j := 0; j < c; j++ {
			result[i][j] = m.At(i, j)
		}
	}
	return result
}

func sliceToDense(data [][]float64) (*mat.Dense, error) {

	r := len(data)
	if r == 0 {
		return nil, fmt.Errorf("Invalid matrix")
	}

	c := len(data[0]) // 看第一個row來判斷有幾個col
	if c == 0 {
		return nil, fmt.Errorf("Invalid matrix")
	}
	result := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++{
		for j := 0; j < c; j ++ {
			 result.Set(i, j, data[i][j])
		}
	}
	return result, nil
}



func SaveModel(nn *NeuralNetwork, filepath string) error {

	hiddenLayers := make([]SerializableLayer, len(nn.Hidden))
	for i :=0; i<len(nn.Hidden); i++{
		hiddenLayers[i] = SerializableLayer{
			Weight: denseToSlice(nn.Hidden[i].weight),
			Bias: denseToSlice(nn.Hidden[i].bias),
		}
	}

	serializableModel := SerializableModel{
		Inputs:       nn.Inputs,
        OutputClass:  nn.OutputClass,
        LearningRate: nn.LearningRate,
		OutputWeight: denseToSlice(nn.OutputWeight),
		OutputBias: denseToSlice(nn.OutputBias),
		HiddenLayers: hiddenLayers,
	}

	jsonData, err := json.MarshalIndent(serializableModel, "", "	")
	if err != nil {
    	return err
	}
	err = os.WriteFile(filepath, jsonData, 0644)
	if err != nil {
    	return err
	}
	return nil
} 




func LoadModel(filename string) (*NeuralNetwork, error){
	jsonData, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	// 解析 JSON 到 SerializableModel
	var serializableModel SerializableModel
	err = json.Unmarshal(jsonData, &serializableModel)
	if err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}

	// 轉換 SerializableModel → NeuralNetwork
	// 轉換 OutputWeight 和 OutputBias
	outputWeight, err := sliceToDense(serializableModel.OutputWeight)
	if err != nil {
		return nil, fmt.Errorf("failed to convert OutputWeight: %w", err)
	}
	outputBias, err := sliceToDense(serializableModel.OutputBias)
	if err != nil {
		return nil, fmt.Errorf("failed to convert OutputBias: %w", err)
	}

	// 轉換 Hidden layers
	hiddenLayers := make([]HiddenLayer, len(serializableModel.HiddenLayers))
	for i, serLayer := range serializableModel.HiddenLayers {
		weight, err := sliceToDense(serLayer.Weight)
		if err != nil {
			return nil, fmt.Errorf("failed to convert hidden layer %d weight: %w", i, err)
		}

		bias, err := sliceToDense(serLayer.Bias)
		if err != nil {
			return nil, fmt.Errorf("failed to convert hidden layer %d bias: %w", i, err)
		}

		hiddenLayers[i] = HiddenLayer{
			nodenum: len(serLayer.Weight),
			weight: weight,
			bias:   bias,
		}
	}

	// 創建 nn
	nn := &NeuralNetwork{
		Inputs:       serializableModel.Inputs,
		OutputClass:  serializableModel.OutputClass,
		Hidden:       hiddenLayers,
		OutputWeight: outputWeight,
		OutputBias:   outputBias,
		LearningRate: serializableModel.LearningRate,
	}

	return nn, nil
}