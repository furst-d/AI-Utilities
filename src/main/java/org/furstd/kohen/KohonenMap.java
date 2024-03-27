package org.furstd.kohen;

import java.util.Arrays;
import java.util.Scanner;

public class KohonenMap {
    private final double[][] weights;
    private final int numNeurons;
    private final int inputDim;

    public KohonenMap(int numNeurons, int inputDim) {
        this.numNeurons = numNeurons;
        this.inputDim = inputDim;
        this.weights = new double[inputDim][numNeurons];
    }

    public void train(double[] input, double alpha, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.println("\nEpoch " + (epoch + 1));

            int bmuIndex = findBMU(input);
            System.out.println("Chosen D" + (bmuIndex + 1));

            for (int j = 0; j < inputDim; j++) {
                weights[j][bmuIndex] += alpha * (input[j] - weights[j][bmuIndex]);
            }

            printWeights();
        }
    }

    public int recall(double[] inputVec) {
        int bmuIndex = findBMU(inputVec);

        StringBuilder coords = new StringBuilder("[");
        for (int i = 0; i < inputDim; i++) {
            coords.append(weights[i][bmuIndex]);
            if (i != inputDim - 1) {
                coords.append(", ");
            }
        }
        coords.append("]");

        System.out.println("BMU Index: " + (bmuIndex + 1) + " with coordinates: " + coords);
        return bmuIndex;
    }

    private int findBMU(double[] input) {
        int bmuIndex = 0;
        double minDist = Double.MAX_VALUE;

        for (int j = 0; j < numNeurons; j++) {
            double dist = 0;
            for (int i = 0; i < inputDim; i++) {
                dist += Math.pow(weights[i][j] - input[i], 2);
            }
            System.out.println("D" + (j + 1) + ": " + dist);

            if (dist < minDist) {
                minDist = dist;
                bmuIndex = j;
            }
        }
        return bmuIndex;
    }

    public void printWeights() {
        System.out.println("Weights:");
        for (double[] weight : weights) {
            System.out.println(Arrays.toString(weight));
        }
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.println("Enter number of neurons:");
        int numNeurons = scanner.nextInt();
        System.out.println("Enter input dimension:");
        int inputDim = scanner.nextInt();

        KohonenMap map = new KohonenMap(numNeurons, inputDim);

        // Požádáme uživatele, aby zadal matici vah
        System.out.println("You must enter the initial weights matrix:");
        for (int i = 0; i < inputDim; i++) {
            System.out.println("Input " + (i + 1) + " weights:");
            for (int j = 0; j < numNeurons; j++) {
                map.weights[i][j] = scanner.nextDouble();
            }
        }


        while (true) {
            System.out.println("\nMenu:");
            System.out.println("1) Train network");
            System.out.println("2) Print weights");
            System.out.println("3) Recall pattern");
            System.out.println("4) Exit");
            System.out.print("Input choice: ");

            int choice = scanner.nextInt();
            switch (choice) {
                case 1:
                    System.out.println("Enter alpha (learning rate):");
                    double alpha = scanner.nextDouble();
                    System.out.println("Enter number of epochs:");
                    int epochs = scanner.nextInt();
                    System.out.println("Enter training patterns separated by spaces (each line a new pattern):");

                    double[] input = new double[inputDim];
                    for (int i = 0; i < inputDim; i++) {
                        input[i] = scanner.nextDouble();
                    }

                    map.train(input, alpha, epochs);
                    break;
                case 2:
                    map.printWeights();
                    break;
                case 3:
                    System.out.println("Input pattern to recall separated by spaces:");
                    double[] testPattern = new double[inputDim];
                    for (int i = 0; i < inputDim; i++) {
                        testPattern[i] = scanner.nextDouble();
                    }
                    map.recall(testPattern);
                    break;
                case 4:
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