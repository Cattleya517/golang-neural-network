package nn

import (
	"fmt"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func main()  {
	m := mat.NewDense(2, 3, nil)
	fmt.Println(m)

	num := rand.Float64()
	fmt.Println(num)
}
