package cmd

import (
	"fmt"
	"os"

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
	case "1. Train self-defined model.":
		fmt.Println("\nTraining mode selected")
		fmt.Println("Define your model structure")
		// TODO: 稍後會調用訓練函數
		
	case "2. Load trained model.":
		fmt.Println("\nLoading model and GUI")
		// TODO: 稍後會調用載入模型函數
		
	case "3. Exit":
		fmt.Println("\nBye！")
		return
		
	default:
		// default 是可選的，處理未預期的情況
		fmt.Println("Unknow Response, Aborting...")
		return
	}
}
