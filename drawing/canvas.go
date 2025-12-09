package drawing

import (
	"fmt"
	"golang-neural-network/nn"
	"image"
	"image/color"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/widget"
	"gonum.org/v1/gonum/mat"
)

// DrawingCanvas 自定義繪圖組件
type DrawingCanvas struct {
	widget.BaseWidget
	img       *image.RGBA
	raster    *canvas.Raster
	drawing   bool
	lastPoint fyne.Position
}

// NewDrawingCanvas 創建新的繪圖畫布
func NewDrawingCanvas(width, height int) *DrawingCanvas {
	dc := &DrawingCanvas{
		img: image.NewRGBA(image.Rect(0, 0, width, height)),
	}
	
	// 初始化為白色背景
	dc.Clear()
	
	// 創建 raster 用於顯示
	dc.raster = canvas.NewRaster(func(w, h int) image.Image {
		return dc.img
	})
	dc.raster.SetMinSize(fyne.NewSize(float32(width), float32(height)))
	
	dc.ExtendBaseWidget(dc)
	return dc
}

// CreateRenderer 實現 Widget 接口
func (dc *DrawingCanvas) CreateRenderer() fyne.WidgetRenderer {
	return widget.NewSimpleRenderer(dc.raster)
}

// Clear 清空畫布
func (dc *DrawingCanvas) Clear() {
	bounds := dc.img.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			dc.img.Set(x, y, color.White)
		}
	}
	dc.raster.Refresh()
}

// Dragged 實現拖曳接口 - 繪製線條
func (dc *DrawingCanvas) Dragged(ev *fyne.DragEvent) {
	if !dc.drawing {
		dc.drawing = true
		dc.lastPoint = ev.Position
		return
	}
	
	// 從上一個點畫線到當前點
	dc.drawLine(dc.lastPoint, ev.Position)
	dc.lastPoint = ev.Position
	dc.raster.Refresh()
}

// DragEnd 拖曳結束
func (dc *DrawingCanvas) DragEnd() {
	dc.drawing = false
}

// Tapped 實現點擊接口 - 繪製單點
func (dc *DrawingCanvas) Tapped(ev *fyne.PointEvent) {
	dc.drawPoint(ev.Position)
	dc.raster.Refresh()
}

// drawLine 畫線（Bresenham 算法）
func (dc *DrawingCanvas) drawLine(from, to fyne.Position) {
	x0, y0 := int(from.X), int(from.Y)
	x1, y1 := int(to.X), int(to.Y)
	
	dx := abs(x1 - x0)
	dy := abs(y1 - y0)
	
	var sx, sy int
	if x0 < x1 {
		sx = 1
	} else {
		sx = -1
	}
	if y0 < y1 {
		sy = 1
	} else {
		sy = -1
	}
	
	err := dx - dy
	
	for {
		dc.drawThickPoint(x0, y0, 12) // 粗筆劃（560/28=20，所以需要約 12-15 像素才能在縮放後保持足夠粗細）
		
		if x0 == x1 && y0 == y1 {
			break
		}
		
		e2 := 2 * err
		if e2 > -dy {
			err -= dy
			x0 += sx
		}
		if e2 < dx {
			err += dx
			y0 += sy
		}
	}
}

// drawPoint 畫單點
func (dc *DrawingCanvas) drawPoint(pos fyne.Position) {
	dc.drawThickPoint(int(pos.X), int(pos.Y), 12)
}

// drawThickPoint 畫粗點（加粗筆畫）
func (dc *DrawingCanvas) drawThickPoint(x, y, thickness int) {
	bounds := dc.img.Bounds()
	for dy := -thickness; dy <= thickness; dy++ {
		for dx := -thickness; dx <= thickness; dx++ {
			px, py := x+dx, y+dy
			if px >= bounds.Min.X && px < bounds.Max.X && 
			   py >= bounds.Min.Y && py < bounds.Max.Y {
				dc.img.Set(px, py, color.Black)
			}
		}
	}
}

// GetImage 獲取當前圖片
func (dc *DrawingCanvas) GetImage() *image.RGBA {
	return dc.img
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// findBoundingBox 找到圖像中非白色像素的邊界框
func findBoundingBox(img *image.RGBA) (minX, minY, maxX, maxY int, found bool) {
	bounds := img.Bounds()
	minX, minY = bounds.Max.X, bounds.Max.Y
	maxX, maxY = bounds.Min.X, bounds.Min.Y
	found = false

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			// 檢查是否為非白色像素（有筆跡）
			if r < 60000 || g < 60000 || b < 60000 {
				found = true
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
				if y < minY {
					minY = y
				}
				if y > maxY {
					maxY = y
				}
			}
		}
	}
	return
}

// resizeBilinear 使用雙線性插值縮放圖片
func resizeBilinear(src *image.RGBA, newWidth, newHeight int) *image.RGBA {
	srcBounds := src.Bounds()
	srcWidth := srcBounds.Dx()
	srcHeight := srcBounds.Dy()

	dst := image.NewRGBA(image.Rect(0, 0, newWidth, newHeight))

	xRatio := float64(srcWidth-1) / float64(newWidth)
	yRatio := float64(srcHeight-1) / float64(newHeight)

	for y := 0; y < newHeight; y++ {
		for x := 0; x < newWidth; x++ {
			srcX := float64(x) * xRatio
			srcY := float64(y) * yRatio

			x0 := int(srcX)
			y0 := int(srcY)
			x1 := x0 + 1
			y1 := y0 + 1

			if x1 >= srcWidth {
				x1 = srcWidth - 1
			}
			if y1 >= srcHeight {
				y1 = srcHeight - 1
			}

			xFrac := srcX - float64(x0)
			yFrac := srcY - float64(y0)

			// 獲取四個角的像素值
			r00, g00, b00, a00 := src.At(x0, y0).RGBA()
			r01, g01, b01, a01 := src.At(x0, y1).RGBA()
			r10, g10, b10, a10 := src.At(x1, y0).RGBA()
			r11, g11, b11, a11 := src.At(x1, y1).RGBA()

			// 雙線性插值
			r := uint8((float64(r00)*(1-xFrac)*(1-yFrac) +
				float64(r10)*xFrac*(1-yFrac) +
				float64(r01)*(1-xFrac)*yFrac +
				float64(r11)*xFrac*yFrac) / 256)
			g := uint8((float64(g00)*(1-xFrac)*(1-yFrac) +
				float64(g10)*xFrac*(1-yFrac) +
				float64(g01)*(1-xFrac)*yFrac +
				float64(g11)*xFrac*yFrac) / 256)
			b := uint8((float64(b00)*(1-xFrac)*(1-yFrac) +
				float64(b10)*xFrac*(1-yFrac) +
				float64(b01)*(1-xFrac)*yFrac +
				float64(b11)*xFrac*yFrac) / 256)
			a := uint8((float64(a00)*(1-xFrac)*(1-yFrac) +
				float64(a10)*xFrac*(1-yFrac) +
				float64(a01)*(1-xFrac)*yFrac +
				float64(a11)*xFrac*yFrac) / 256)

			dst.Set(x, y, color.RGBA{r, g, b, a})
		}
	}

	return dst
}

// preprocessForMNIST 將手寫圖片預處理成 MNIST 格式（居中 + 縮放到 20x20 並放在 28x28 中心）
func preprocessForMNIST(src *image.RGBA) *image.RGBA {
	// 1. 找到數字的邊界框
	minX, minY, maxX, maxY, found := findBoundingBox(src)
	if !found {
		// 沒有找到任何筆跡，返回空白圖片
		return image.NewRGBA(image.Rect(0, 0, 28, 28))
	}

	// 2. 裁剪出數字區域
	digitWidth := maxX - minX + 1
	digitHeight := maxY - minY + 1

	cropped := image.NewRGBA(image.Rect(0, 0, digitWidth, digitHeight))
	for y := 0; y < digitHeight; y++ {
		for x := 0; x < digitWidth; x++ {
			cropped.Set(x, y, src.At(minX+x, minY+y))
		}
	}

	// 3. 計算縮放比例，保持長寬比，放入 20x20 區域
	targetSize := 20.0
	scale := targetSize / float64(max(digitWidth, digitHeight))

	newWidth := int(float64(digitWidth) * scale)
	newHeight := int(float64(digitHeight) * scale)

	if newWidth < 1 {
		newWidth = 1
	}
	if newHeight < 1 {
		newHeight = 1
	}

	// 4. 縮放數字
	resized := resizeBilinear(cropped, newWidth, newHeight)

	// 5. 創建 28x28 白色背景圖片，將數字放在中心
	result := image.NewRGBA(image.Rect(0, 0, 28, 28))
	// 初始化為白色
	for y := 0; y < 28; y++ {
		for x := 0; x < 28; x++ {
			result.Set(x, y, color.White)
		}
	}

	// 計算偏移量使數字居中
	offsetX := (28 - newWidth) / 2
	offsetY := (28 - newHeight) / 2

	// 複製縮放後的數字到中心
	for y := 0; y < newHeight; y++ {
		for x := 0; x < newWidth; x++ {
			result.Set(offsetX+x, offsetY+y, resized.At(x, y))
		}
	}

	return result
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// imageToGrayscaleMatrix 將圖片轉為灰階並標準化到 0-1，返回 784x1 矩陣
func imageToGrayscaleMatrix(img *image.RGBA) *mat.Dense {
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()
	
	// 創建 784x1 的矩陣（28*28 = 784）
	data := make([]float64, width*height)
	
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			// 獲取像素
			r, g, b, _ := img.At(x, y).RGBA()
			
			// 轉為灰階（0-65535 範圍）
			// 白色背景(255,255,255) -> 65535，黑色筆跡(0,0,0) -> 0
			gray := (r + g + b) / 3
			
			// 反轉：MNIST 是黑底白字，user畫的是白底黑字
			// 標準化到 0-1：黑色筆跡 -> 1.0，白色背景 -> 0.0
			normalized := 1.0 - float64(gray)/65535.0
			
			// 存入矩陣（按行優先順序）
			data[y*width+x] = normalized
		}
	}
	
	return mat.NewDense(width*height, 1, data)
}


// ShowDrawingBoard 顯示手寫板界面
func ShowDrawingBoard() {
	model, err := nn.LoadModel("/Users/user/github-projects/golang-neural-network/models/basic.json")
	if err != nil {
		fmt.Printf("Error loading model: %v\n", err)
		fmt.Println("Launching drawing board without model...")
		ShowDrawingBoardWithModel(nil)
		return
	}
	fmt.Println("Model loaded successfully!")
	ShowDrawingBoardWithModel(model)
}

// ShowDrawingBoardWithModel 顯示手寫板界面（可選模型）
func ShowDrawingBoardWithModel(model *nn.NeuralNetwork) {
	a := app.New()
	w := a.NewWindow("MNIST Drawing Board - 28x28 Digit Recognition")

	// 創建 280x280 的畫布（會縮放到 28x28）
	drawingCanvas := NewDrawingCanvas(560, 560)
	
	// 結果標籤
	resultLabel := widget.NewLabel("Draw a digit (0-9) and click Predict")
	
	// 清除按鈕
	clearBtn := widget.NewButton("Clear", func() {
		drawingCanvas.Clear()
		resultLabel.SetText("Draw a digit (0-9) and click Predict")
	})
	
	// 預測按鈕
	predictBtn := widget.NewButton("Predict", func() {
		if model == nil {
			resultLabel.SetText("Error: No model loaded")
			return
		}

		// 1. 預處理：找到數字邊界、居中、縮放到 20x20 並放在 28x28 中心
		preprocessed := preprocessForMNIST(drawingCanvas.GetImage())

		// 2. 轉為灰階並標準化（784x1 矩陣）
		inputMatrix := imageToGrayscaleMatrix(preprocessed)
		
		// 3. 模型預測
		logits, err := model.Forward(inputMatrix)
		if err != nil {
			resultLabel.SetText(fmt.Sprintf("Prediction Error: %v", err))
			return
		}
		
		// softmax 獲得概率分佈
		probs := nn.Softmax(logits)
		
		// 5. argmax最大值索引 
		prediction, err := nn.Argmax(probs)
		if err != nil {
			resultLabel.SetText(fmt.Sprintf("Argmax Error: %v", err))
			return
		}
		
		// 顯示結果
		maxValue := probs.At(prediction, 0)
		confidence := maxValue * 100
		resultLabel.SetText(fmt.Sprintf("Prediction: %d (Confidence: %.1f%%)", prediction, confidence))
	})
	
	// layout
	buttons := container.NewHBox(clearBtn, predictBtn)
	content := container.NewBorder(
		nil,                    // top
		container.NewVBox(buttons, resultLabel), // bottom
		nil,                    // left
		nil,                    // right
		drawingCanvas,          // center
	)
	
	w.SetContent(content)
	w.Resize(fyne.NewSize(600, 700))
	w.ShowAndRun()
}