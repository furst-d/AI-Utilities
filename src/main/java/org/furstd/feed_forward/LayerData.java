package org.furstd.feed_forward;

public class LayerData {
    private int neuronCount;

    private double[] y;

    private ActivationFunction activationFunction;

    private double[] localGradients;

    public int getNeuronCount() {
        return neuronCount;
    }

    public void setNeuronCount(int neuronCount) {
        this.neuronCount = neuronCount;
    }

    public double[] getY() {
        return y;
    }

    public void setY(double[] y) {
        this.y = y;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public double[] getLocalGradients() {
        return localGradients;
    }

    public void setLocalGradients(double[] localGradients) {
        this.localGradients = localGradients;
    }
}
