package nn

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type NeuralNetwork struct {
	Inputs       int
	Hidden       []HiddenLayer 
	OutputClass  int
	OutputWeight *mat.Dense  
    OutputBias   *mat.Dense 
	LearningRate float64
}

type HiddenLayer struct{
	nodenum int
	weight *mat.Dense
	bias *mat.Dense
}

func randomArray(size int) []float64 {
	array := make([]float64, size)
	for i := 0; i < size; i++ {
		array[i] = rand.Float64() - 0.5
	}
	return array
}

func NewNeuralNetwork(inputs, outputClass int, hiddenNodes []int, learningRate float64) (*NeuralNetwork, error){
	// Let user define the nn structure 
	nn := &NeuralNetwork{
		Inputs: inputs,
		OutputClass: outputClass,
		LearningRate: learningRate,
		Hidden: make([]HiddenLayer, 0, len(hiddenNodes)),
	}

	for idx, node := range hiddenNodes {
		var hidden HiddenLayer
		if idx == 0 {
			hidden = HiddenLayer{
			nodenum: node,
			weight: mat.NewDense(node, inputs, randomArray(node*inputs)),
			bias: mat.NewDense(node, 1, randomArray(node)),
			}
		} else {
			hidden = HiddenLayer{
				nodenum: node,
				weight:  mat.NewDense(node, hiddenNodes[idx-1], randomArray(node*hiddenNodes[idx-1])),
				bias:    mat.NewDense(node, 1, randomArray(node)),
			}
		}
		nn.Hidden = append(nn.Hidden, hidden)
	}

	nn.OutputWeight = mat.NewDense(outputClass, hiddenNodes[len(hiddenNodes)-1], randomArray(outputClass*hiddenNodes[len(hiddenNodes)-1]))
	nn.OutputBias = mat.NewDense(outputClass, 1, randomArray(outputClass))
	return nn, nil
}

/*
nn := NewNeuralNetwork(
    10,              // 輸入 10 個特徵
    3,               // 輸出 3 個類別
    []int{64, 32},   // 兩個隱藏層：64 和 32 個節點
    0.01,            // 學習率
)
*/

func relu(m *mat.Dense) *mat.Dense{
	r, c := m.Dims()
	new_matrix := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++{
		for j := 0; j < c; j++ {
			value := m.At(i, j)
			new_value := max(value, 0)
			new_matrix.Set(i, j, new_value)
		}
	}
	return new_matrix
}


func (nn NeuralNetwork) forward(input *mat.Dense) (*mat.Dense, error) {
	current := input
	// process through hidden layer
	for _, layer := range nn.Hidden {

		weighted := mat.Dense{}
		weighted.Mul(layer.weight, current)
		weighted.Add(&weighted, layer.bias)

		// activation function
		activated := relu(&weighted)

		current = activated
	}
	// process through output layer
	output := mat.Dense{}
	output.Mul(nn.OutputWeight, current)
	output.Add(&output, nn.OutputBias)

	return &output, nil
}






