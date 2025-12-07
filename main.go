package main

import (
	"fmt"
	"golang-neural-network/nn"
	"log"
)

func main(){
	// transfer training set from byte file to csv
	err := nn.ConvertToCSV(
        "mnist_data/train-images.idx3-ubyte",
        "mnist_data/train-labels.idx1-ubyte",
        "mnist_data/train.csv",
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("訓練集轉換完成!")
	// transfer testing set from byte file to csv
	err = nn.ConvertToCSV(
        "mnist_data/t10k-images.idx3-ubyte",
        "mnist_data/t10k-labels.idx1-ubyte",
        "mnist_data/test.csv",
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("測試集轉換完成!")
}