package org.furstd.convolutional;

import org.furstd.kohen.KohonenMap;

import java.util.Scanner;

public class ConvolutionalNetwork {

    public void convPlusMaxPooling(double[][] input, double[][] filter) {
        int inputWidth = input.length;
        int inputHeight = input[0].length;
        int filterWidth = filter.length;
        int filterHeight = filter[0].length;

        int outputWidth = inputWidth - filterWidth + 1;
        int outputHeight = inputHeight - filterHeight + 1;

        double[][] output = new double[outputWidth][outputHeight];

        for (int i = 0; i < outputWidth; i++) {
            for (int j = 0; j < outputHeight; j++) {
                double sum = 0;
                for (int k = 0; k < filterWidth; k++) {
                    for (int l = 0; l < filterHeight; l++) {
                        sum += input[i + k][j + l] * filter[k][l];
                    }
                }
                output[i][j] = sum;
            }
        }

        System.out.println("\nOutput matrix after convolution:");
        printMatrix(output);

        System.out.println("\nOutput matrix after max pooling:");
        maxPooling(output, outputWidth, outputHeight);
    }

    private void maxPooling(double[][] input, int poolWidth, int poolHeight) {
        int inputWidth = input.length;
        int inputHeight = input[0].length;

        int outputWidth = inputWidth / poolWidth;
        int outputHeight = inputHeight / poolHeight;

        double[][] output = new double[poolWidth][poolHeight];

        for (int i = 0; i < outputWidth; i++) {
            for (int j = 0; j < outputHeight; j++) {
                double max = input[i * poolWidth][j * poolHeight];
                for (int k = 0; k < poolWidth; k++) {
                    for (int l = 0; l < poolHeight; l++) {
                        if (input[i * poolWidth + k][j * poolHeight + l] > max) {
                            max = input[i * poolWidth + k][j * poolHeight + l];
                        }
                    }
                }
                output[i][j] = max;
            }
        }

        printMatrix(output);
    }

    public void printMatrix(double[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                System.out.print(matrix[i][j] + " ");
            }
            System.out.println();
        }
    }

    private double findMax(double[][] matrix) {
        double max = matrix[0][0];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (matrix[i][j] > max) {
                    max = matrix[i][j];
                }
            }
        }
        return max;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        ConvolutionalNetwork cnn = new ConvolutionalNetwork();

        while (true) {
            System.out.println("\nMenu:");
            System.out.println("1) Convolution + Max Pooling");
            System.out.println("2) Max Pooling");
            System.out.println("3) Exit");
            System.out.print("Input choice: ");

            int choice = scanner.nextInt();
            switch (choice) {
                case 1:
                    System.out.println("Enter input matrix size (width x height) separated by space:");
                    double[][] inputMatrix = new double[scanner.nextInt()][scanner.nextInt()];
                    for (int i = 0; i < inputMatrix[0].length; i++) {
                        System.out.println("Enter row " + (i + 1) + " of input matrix separated by spaces:");
                        for (int j = 0; j < inputMatrix.length; j++) {
                            inputMatrix[i][j] = scanner.nextDouble();
                        }
                    }

                    System.out.println("Enter filter size (width x height) separated by space:");
                    double[][] filter = new double[scanner.nextInt()][scanner.nextInt()];
                    for (int i = 0; i < filter[0].length; i++) {
                        System.out.println("Enter row " + (i + 1) + " of filter matrix separated by spaces:");
                        for (int j = 0; j < filter.length; j++) {
                            filter[i][j] = scanner.nextDouble();
                        }
                    }

                    cnn.convPlusMaxPooling(inputMatrix, filter);
                    break;
                case 2:
                    System.out.println("Enter input matrix size (width x height) separated by space:");
                    inputMatrix = new double[scanner.nextInt()][scanner.nextInt()];
                    for (int i = 0; i < inputMatrix[0].length; i++) {
                        System.out.println("Enter row " + (i + 1) + " of input matrix separated by spaces:");
                        for (int j = 0; j < inputMatrix.length; j++) {
                            inputMatrix[i][j] = scanner.nextDouble();
                        }
                    }

                    System.out.println("Enter pool size (width x height) separated by space:");
                    int poolWidth = scanner.nextInt();
                    int poolHeight = scanner.nextInt();

                    cnn.maxPooling(inputMatrix, poolWidth, poolHeight);
                    break;
                case 3:
                    System.out.println("Exiting...");
                    scanner.close();
                    return;
                default:
                    System.out.println("Invalid choice!");
                    break;
            }
        }
    }
}
