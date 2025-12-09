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

// He initialization for ReLU activation
// stddev = sqrt(2 / n_inputs)
func heInitArray(size int, nInputs int) []float64 {
	array := make([]float64, size)
	stddev := math.Sqrt(2.0 / float64(nInputs))
	for i := 0; i < size; i++ {
		// Box-Muller transform for normal distribution
		// Avoid u1=0 which would cause log(0)=-Inf
		u1 := rand.Float64()
		for u1 == 0 {
			u1 = rand.Float64()
		}
		u2 := rand.Float64()
		z := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
		array[i] = z * stddev
	}
	return array
}

func zeroBiasArray(size int) []float64 {
	return make([]float64, size) // initialized to zeros
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
			weight: mat.NewDense(node, inputs, heInitArray(node*inputs, inputs)),
			bias: mat.NewDense(node, 1, zeroBiasArray(node)),
			}
		} else {
			hidden = HiddenLayer{
				nodenum: node,
				weight:  mat.NewDense(node, hiddenNodes[idx-1], heInitArray(node*hiddenNodes[idx-1], hiddenNodes[idx-1])),
				bias:    mat.NewDense(node, 1, zeroBiasArray(node)),
			}
		}
		nn.Hidden = append(nn.Hidden, hidden)
	}

	lastHiddenSize := hiddenNodes[len(hiddenNodes)-1]
	nn.OutputWeight = mat.NewDense(outputClass, lastHiddenSize, heInitArray(outputClass*lastHiddenSize, lastHiddenSize))
	nn.OutputBias = mat.NewDense(outputClass, 1, zeroBiasArray(outputClass))
	return nn, nil
}


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


func (nn NeuralNetwork) Forward(input *mat.Dense) (*mat.Dense, error) {
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


func Softmax(input *mat.Dense) *mat.Dense {
	max := input.At(0, 0)
	r, c := input.Dims()
	for i := 0; i < r; i++{
		for j:=0; j < c; j++{
			max = math.Max(max, input.At(i, j))
		}
	}
	result := mat.NewDense(r, c, nil)
	sum := 0.0
	for i := 0; i < r; i++{
		for j:=0; j < c; j++{
			new := math.Exp(input.At(i, j) - max)
			result.Set(i, j, new)
			sum += new
		}
	}
	for i := 0; i < r; i++{
		for j:=0; j < c; j++{
			new := result.At(i, j) / sum
			result.Set(i, j, new)
		}
	}
	return result
}

