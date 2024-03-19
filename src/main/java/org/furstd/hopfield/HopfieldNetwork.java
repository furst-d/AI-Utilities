package org.furstd.hopfield;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class HopfieldNetwork {
    private int[][] weightMatrix;
    private final int size;
    private final ArrayList<int[]> patterns = new ArrayList<>();

    public HopfieldNetwork(int size) {
        this.size = size;
        this.weightMatrix = new int[size][size];
    }

    public void addPattern(int[] pattern) {
        if (pattern.length != size) {
            throw new IllegalArgumentException("Pattern must have size " + size);
        }
        patterns.add(pattern);
        updateWeights();
    }

    private void updateWeights() {
        // Reset váhové matice
        for (int i = 0; i < size; i++) {
            Arrays.fill(weightMatrix[i], 0);
        }

        // Výpočet váh podle akumulace vzorů
        if (patterns.size() >= 2) {
            int[] pattern1 = patterns.get(0);
            int[] pattern2 = patterns.get(1);

            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    if (i != j) {
                        // Akumulace pouze pro první dva vzory podle vašeho výpočtu
                        weightMatrix[i][j] = pattern1[i] * pattern2[j] + pattern2[i] * pattern1[j];
                    }
                }
            }
        }
    }

    public void setWeightsDirectly(int[][] newWeights) {
        if (newWeights.length != size || newWeights[0].length != size) {
            throw new IllegalArgumentException("Weight matrix must have size " + size + "x" + size);
        }
        this.weightMatrix = newWeights;
    }

    public int[] recall(int[] pattern) {
        if (pattern.length != size) {
            throw new IllegalArgumentException("Pattern must have size " + size);
        }
        int[] recalledPattern = Arrays.copyOf(pattern, size);
        int[] newPattern = Arrays.copyOf(recalledPattern, recalledPattern.length);
        boolean stable;
        int iteration = 0;
        do {
            System.out.println("Iteration " + (iteration + 1) + ": " + Arrays.toString(recalledPattern));
            stable = true;
            for (int i = 0; i < size; i++) {
                int sum = 0;
                for (int j = 0; j < size; j++) {
                    sum += weightMatrix[i][j] * recalledPattern[j];
                }
                int updatedValue = sum >= 0 ? 1 : -1;
                System.out.printf("y%d = f(%d) = %d%n", i + 1, sum, updatedValue);
                if (updatedValue != recalledPattern[i]) {
                    newPattern[i] = updatedValue;
                    stable = false;
                }
            }
            recalledPattern = Arrays.copyOf(newPattern, newPattern.length);
            System.out.println();
            iteration++;
        } while (!stable);
        return recalledPattern;
    }

    public void printMatrix() {
        for (int[] row : weightMatrix) {
            for (int val : row) {
                System.out.printf("%4d", val);
            }
            System.out.println();
        }
        System.out.println();
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.println("Input vector size: ");
        int size = scanner.nextInt();

        HopfieldNetwork network = new HopfieldNetwork(size);

        while (true) {
            System.out.println("\nMenu:");
            System.out.println("1) Add patterns");
            System.out.println("2) Set weight matrix directly");
            System.out.println("3) Print weight matrix");
            System.out.println("4) Recall pattern");
            System.out.println("5) Exit");
            System.out.print("Input choice: ");

            int choice = scanner.nextInt();
            switch (choice) {
                case 1:
                    for (int i = 0; i < 2; i++) {
                        System.out.println("Input " + (i == 0 ? "first" : "second") + " pattern separated by spaces:");
                        int[] pattern = new int[size];
                        for (int j = 0; j < size; j++) {
                            pattern[j] = scanner.nextInt();
                        }
                        network.addPattern(pattern);
                    }
                    break;
                case 2:
                    System.out.println("Input weight matrix separated by spaces (one line):");
                    int[][] weights = new int[size][size];
                    for (int i = 0; i < size; i++) {
                        for (int j = 0; j < size; j++) {
                            weights[i][j] = scanner.nextInt();
                        }
                    }
                    network.setWeightsDirectly(weights);
                    break;
                case 3:
                    network.printMatrix();
                    break;
                case 4:
                    System.out.println("Input pattern to recall separated by spaces:");
                    int[] testPattern = new int[size];
                    for (int i = 0; i < size; i++) {
                        testPattern[i] = scanner.nextInt();
                    }
                    int[] recalledPattern = network.recall(testPattern);
                    System.out.println("Recalled pattern: " + Arrays.toString(recalledPattern));
                    break;
                case 5:
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

