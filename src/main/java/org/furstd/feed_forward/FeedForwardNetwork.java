package org.furstd.feed_forward;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class FeedForwardNetwork {
    private double[][][] weights;
    private final List<Double> x;
    private final List<LayerData> layersData;

    public FeedForwardNetwork() {
        x = new ArrayList<>();
        layersData = new ArrayList<>();
    }

    public void addX(double x) {
        this.x.add(x);
    }

    public void addLayerData(LayerData layerData) {
        layersData.add(layerData);
    }

    public int getNumberOfInputs() {
        return weights[0].length - 1; // -1 protože bias
    }

    public void setWeights(double[][][] weights) {
        this.weights = weights;
    }

    private int[] loadNeuronsPerLayer(Scanner scanner, int numberOfLayers) {
        System.out.println("Enter number of neurons for each layer (separated by space): ");
        int[] neuronPerLayer = new int[numberOfLayers];
        for (int i = 0; i < numberOfLayers; i++) {
            neuronPerLayer[i] = scanner.nextInt();
            if (i > 0) {
                LayerData layerData = new LayerData();
                layerData.setNeuronCount(neuronPerLayer[i]);
                layersData.add(layerData);
            }
        }
        return neuronPerLayer;
    }

    private void loadActivationFunctions(Scanner scanner, int numberOfLayers) {
        for (int i = 0; i < numberOfLayers - 1; i++) {
            LayerData layerData = layersData.get(i);
            System.out.println("\nChoose activation function for layer " + (i + 1));
            System.out.println("1) Hyperbolic tangent - y = tanh(ya)");
            System.out.println("2) Linear ident - y = ya");
            System.out.print("Input choice: ");
            int choice = scanner.nextInt();
            switch (choice) {
                case 1:
                    layerData.setActivationFunction(ActivationFunction.HYPERBOLIC_TANGENT);
                    break;
                case 2:
                    layerData.setActivationFunction(ActivationFunction.LINEAR_IDENT);
                    break;
                default:
                    System.out.println("Invalid choice!");
                    break;
            }
        }
    }

    public void initializeNetwork(int numberOfLayers, Scanner scanner) {
        int[] neuronPerLayer = loadNeuronsPerLayer(scanner, numberOfLayers);
        loadActivationFunctions(scanner, numberOfLayers);

        weights = new double[numberOfLayers - 1][][];

        for (int i = 0; i < numberOfLayers - 1; i++) {
            int currentLayerNeuronCount = neuronPerLayer[i];
            int nextLayerNeuronCount = neuronPerLayer[i + 1];

            weights[i] = new double[currentLayerNeuronCount + 1][nextLayerNeuronCount];

            for (int j = 0; j < currentLayerNeuronCount + 1; j++) { // + 1 protože bias
                System.out.println("Enter weights for " + (j == 0 ? "bias" : "neuron " + j) + " in layer " + i + " to all neurons in the layer: ");
                for (int k = 0; k < nextLayerNeuronCount; k++) {
                    weights[i][j][k] = scanner.nextDouble();
                }
            }
        }
    }

    public void printWeights(double[][][] weights) {
        for (int layer = 0; layer < weights.length; layer++) {
            System.out.println("Weights of layer w" + (layer + 1) + ":");

            for (int neuron = 0; neuron < weights[layer].length; neuron++) {
                for (int input = 0; input < weights[layer][0].length; input++) {
                    System.out.print(String.format("%.4f ", weights[layer][neuron][input]).replace(",", "."));
                }
                System.out.println();
            }
            System.out.println();
        }
    }

    public double roundTo4DecimalPlaces(double value) {
        return Math.round(value * 10000.0) / 10000.0;
    }

    private double[] addBiasToInput(double[] input) {
        double[] inputWithBias = new double[input.length + 1];
        inputWithBias[0] = 1; // Bias
        System.arraycopy(input, 0, inputWithBias, 1, input.length);
        return inputWithBias;
    }

    private double[][] transposeMatrix(double[][] weights) {
        double[][] transposed = new double[weights[0].length][weights.length];

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                transposed[j][i] = weights[i][j];
            }
        }

        return transposed;
    }

    private double[] applyActivationFunction(double[] input, ActivationFunction activationFunction) {
        switch (activationFunction) {
            case HYPERBOLIC_TANGENT:
                double[] output = new double[input.length];
                for (int i = 0; i < input.length; i++) {
                    output[i] = roundTo4DecimalPlaces(Math.tanh(input[i]));
                }
                return output;
            case LINEAR_IDENT:
                return input;
            default:
                throw new IllegalArgumentException("Unknown activation function");
        }
    }

    public void computeResponse() {
        double[] input = x.stream().mapToDouble(Double::doubleValue).toArray();

        for (int i = 0; i < weights.length; i++) {
            LayerData layerData = layersData.get(i);
            input = addBiasToInput(input);

            double[][] transposeMatrix = transposeMatrix(weights[i]);
            double[] output = new double[transposeMatrix.length];

            for (int j = 0; j < transposeMatrix.length; j++) {
                for (int k = 0; k < transposeMatrix[0].length; k++) {
                    double value = transposeMatrix[j][k];
                    output[j] += input[k] * value;
                }
                output[j] = roundTo4DecimalPlaces(output[j]);
            }

            System.out.println("\ny" + (i + 1) + "a = " + Arrays.toString(output));
            input = applyActivationFunction(output, layerData.getActivationFunction());
            layerData.setY(input);
            System.out.println("y" + (i + 1) + " = " + Arrays.toString(input));
        }
    }

    public void computeBGD(double[] t, double alpha) {
        double[] e = new double[t.length];
        double[] finalY = layersData.get(layersData.size() - 1).getY();

        /*
        Compute error vector e
         */
        for (int i = 0; i < t.length; i++) {
            e[i] = roundTo4DecimalPlaces(t[i] - finalY[i]);
        }
        System.out.println("Error vector e: " + Arrays.toString(e));

        /*
        Compute local gradients
        */

        for (int layer = weights.length - 1; layer >= 0; layer--) {
            LayerData layerData = layersData.get(layer);
            double[] localGradients = new double[layerData.getNeuronCount()];

            for (int neuron = 0; neuron < layerData.getNeuronCount(); neuron++) {
                if (layerData.getActivationFunction() == ActivationFunction.LINEAR_IDENT) {
                    localGradients[neuron] = e[neuron];
                } else {
                    double derivation = roundTo4DecimalPlaces(derivateTanh(layerData.getY()[neuron]));
                    LayerData nextLayerData = layersData.get(layer + 1);
                    double sum = 0;
                    for (int i = 0; i < layerData.getNeuronCount(); i++) {
                        double[] neuronWeights = weights[layer + 1][neuron + 1];
                        sum += nextLayerData.getLocalGradients()[i] * neuronWeights[i];
                    }

                    double delta = derivation * sum;
                    localGradients[neuron] = roundTo4DecimalPlaces(delta);
                }
            }
            layerData.setLocalGradients(localGradients);
            System.out.println("Local gradients for layer " + layer + ": " + Arrays.toString(localGradients));
        }

        /*
        Compute new weights
         */
        double[][][] backpropagationWeights = new double[weights.length][][];
        for (int layer = weights.length - 1; layer >= 0; layer--) {
            backpropagationWeights[layer] = new double[weights[layer].length][weights[layer][0].length];
            LayerData layerData = layersData.get(layer);

            double[] previousLayerYVector = getYVector(layer - 1);
            for (int neuron = 0; neuron < layerData.getNeuronCount(); neuron++) {
                backpropagationWeights[layer][neuron] = new double[layerData.getNeuronCount()];
                for (int i = 0; i < layerData.getNeuronCount(); i++) {
                    double oldWeight = weights[layer][neuron][i];
                    double delta = layerData.getLocalGradients()[i];
                    double backpropagation = roundTo4DecimalPlaces(alpha * delta * previousLayerYVector[neuron]);
                    backpropagationWeights[layer][neuron][i] = backpropagation;

                    double newWeight = oldWeight + backpropagation;
                    weights[layer][neuron][i] = newWeight;
                }
            }
        }

        System.out.println("\nBackpropagation weights:");
        printWeights(backpropagationWeights);
        System.out.println("\nNew weights:");
        printWeights(weights);
    }

    private double derivateTanh(double value) {
        return 1 - Math.pow(value, 2);
    }

    private double[] getYVector(int layer) {
        if (layer == -1) {
            return addBiasToInput(x.stream().mapToDouble(Double::doubleValue).toArray());
        } else {
            return addBiasToInput(layersData.get(layer).getY());
        }
    }


    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        FeedForwardNetwork ffnn = new FeedForwardNetwork();

        System.out.println("Enter number of layers (included input layer): ");
        int numberOfLayers = scanner.nextInt();

        ffnn.initializeNetwork(numberOfLayers, scanner);

        // Pro účely testování matici sestavíme ručně
//        double weights[][][] = {
//                {
//                        {0.8, 0.3},
//                        {-0.1, 1.2}
//                },
//                {
//                        {-0.1, 0.2},
//                        {0.4, 0.3},
//                        {-0.4, 0.8}
//                }
//        };
//
//        LayerData l1 = new LayerData();
//        l1.setNeuronCount(2);
//        l1.setActivationFunction(ActivationFunction.HYPERBOLIC_TANGENT);
//        ffnn.addLayerData(l1);
//        LayerData l2 = new LayerData();
//        l2.setNeuronCount(2);
//        l2.setActivationFunction(ActivationFunction.LINEAR_IDENT);
//        ffnn.addLayerData(l2);
//        ffnn.setWeights(weights);

//        double weights[][][] = {
//                {
//                        {1, -1},
//                        {1, 2}
//                },
//                {
//                        {1, 1},
//                        {2, -1},
//                        {1, 2}
//                }
//        };
//
//        LayerData l1 = new LayerData();
//        l1.setNeuronCount(2);
//        l1.setActivationFunction(ActivationFunction.HYPERBOLIC_TANGENT);
//        ffnn.addLayerData(l1);
//        LayerData l2 = new LayerData();
//        l2.setNeuronCount(2);
//        l2.setActivationFunction(ActivationFunction.LINEAR_IDENT);
//        ffnn.addLayerData(l2);
//        ffnn.setWeights(weights);


//        double weights[][][] = {
//                {
//                        {0.5, -0.5},
//                        {1, 2},
//                        {2, 1}
//                },
//                {
//                        {0.5},
//                        {0.5},
//                        {0.5}
//                }
//        };
//        LayerData l1 = new LayerData();
//        l1.setNeuronCount(2);
//        l1.setActivationFunction(ActivationFunction.LINEAR_IDENT);
//        ffnn.addLayerData(l1);
//        LayerData l2 = new LayerData();
//        l2.setNeuronCount(1);
//        l2.setActivationFunction(ActivationFunction.LINEAR_IDENT);
//        ffnn.addLayerData(l2);
//        ffnn.setWeights(weights);

        while (true) {
            System.out.println("\nMenu:");
            System.out.println("1) Compute response");
            System.out.println("2) Print weights");
            System.out.println("3) Compute BGD");
            System.out.println("4) Exit");
            System.out.print("Input choice: ");

            int choice = scanner.nextInt();
            switch (choice) {
                case 1:
                    System.out.println("Enter input vector separated by spaces: ");
                    double[] input = new double[ffnn.getNumberOfInputs()];
                    for (int i = 0; i < input.length; i++) {
                        ffnn.addX(scanner.nextDouble());
                    }
                    ffnn.computeResponse();
                    break;
                case 2:
                    ffnn.printWeights(ffnn.weights);
                    break;
                case 3:
//                    ffnn.addX(0.25);
//                    l1.setY(new double[]{0.649, 0.537});
//                    l2.setY(new double[]{-0.0549, 0.8246});

                    int tSize = ffnn.layersData.get(ffnn.layersData.size() - 1).getNeuronCount();
                    System.out.println("Enter t vector separated by spaces: ");
                    double[] t = new double[tSize];
                    for (int i = 0; i < t.length; i++) {
                        t[i] = scanner.nextDouble();
                    }

                    System.out.println("Enter alpha: ");
                    double alpha = scanner.nextDouble();

                    ffnn.computeBGD(t, alpha);
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
