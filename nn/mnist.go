package nn

import (
	"encoding/binary"
	"fmt"
	"os"
)

func ReadMNISTImages(filename string) ([][]byte, error){
	file, err := os.Open(filename)
	if err != nil{
		return nil, fmt.Errorf("Can't open file %s", filename)
	}
	defer file.Close()
	// 1. 創建一個16字節的buffer來存header
	header := make([]byte, 16)

	// 2. 讀取16字節到buffer
	_, err = file.Read(header)

	// 3. 從header中解析數字 (Big Endian格式)
	numImages := binary.BigEndian.Uint32(header[4:8])  // 圖片數量
	rows := binary.BigEndian.Uint32(header[8:12])      // 行數
	cols := binary.BigEndian.Uint32(header[12:16])     // 列數

	images := make([][]byte, numImages)
	// [][]byte -> 一個存放多個 []byte 的slice (代表多張圖片)
	// 第一個slice存入圖片編號，第二個slice存入像素編號

	// 將每張圖片存進slice
	for i := 0; i < int(numImages); i++{
		imageSize := rows * cols
		image := make([]byte, imageSize)

		// 讀取這張圖片的所有像素
		_, err = file.Read(image)
		if err != nil {
			return nil, fmt.Errorf("failed to read header: %v", err)
		}
		images[i] = image
	}
	return images, nil
}

func ReadMNISTLabels(filename string) ([]byte, error){
	file, err := os.Open(filename)
	if err != nil{
		return nil, fmt.Errorf("Can't open file %s", filename)
	}
	defer file.Close()
	header := make([]byte, 8)

	_, err = file.Read(header)
	numLabels := binary.BigEndian.Uint32(header[4:8])

	labels := make([]byte, numLabels)

	_, err = file.Read(labels)
	if err != nil {
		return nil, fmt.Errorf("failed to read labels: %v", err)
	}
	return labels, nil
}

// ConvertToCSV 將MNIST數據轉換為CSV格式
// CSV格式: 每行為 label,pixel1,pixel2,...,pixel784
func ConvertToCSV(imagesFile, labelsFile, outputFile string) error {
	// 讀取圖片和標籤
	images, err := ReadMNISTImages(imagesFile)
	if err != nil {
		return fmt.Errorf("failed to read images: %v", err)
	}

	labels, err := ReadMNISTLabels(labelsFile)
	if err != nil {
		return fmt.Errorf("failed to read labels: %v", err)
	}

	// 檢查圖片和標籤數量是否匹配
	if len(images) != len(labels) {
		return fmt.Errorf("number of images (%d) doesn't match number of labels (%d)", len(images), len(labels))
	}

	// 創建CSV文件
	file, err := os.Create(outputFile)
	if err != nil {
		return fmt.Errorf("failed to create output file: %v", err)
	}
	defer file.Close()

	// 寫入每一行數據
	for i := 0; i < len(images); i++ {
		// 寫入標籤
		fmt.Fprintf(file, "%d", labels[i])

		// 寫入所有像素值
		for _, pixel := range images[i] {
			fmt.Fprintf(file, ",%d", pixel)
		}

		// 換行
		fmt.Fprintln(file)
	}

	return nil
}

