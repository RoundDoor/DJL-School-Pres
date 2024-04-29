package org.example;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.engine.Engine;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;


/**
 * This class is used to train a Multi-Layer Perceptron (MLP) model on the MNIST dataset.
 */
public class Trainer {
    public static void main(String[] args) throws TranslateException, IOException {

        trainer();

    }

    /**
     * This method is used to train the MLP model.
     * It first creates the MLP model, then loads the MNIST dataset.
     * After that, it creates a training configuration and a trainer.
     * Finally, it trains the model and saves it.
     */
    public static void trainer() throws TranslateException, IOException {

        // Create a Neural Network
        Block block =
                new Mlp(
                        // Define the input size of the neural network
                        Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH,
                        // Define the number of classes in the dataset
                        Mnist.NUM_CLASSES,
                        // Define the number of neurons in each layer of the neural network
                        // 128 in the first hidden layer and 64 in the second hidden layer
                        new int[]{128, 64});

        // Create a new model
        Model model = Model.newInstance("mlp");
        model.setBlock(block);

        // Load the dataset
        // With a batch size of 1024 and random sampling of the dataset
        RandomAccessDataset mnistTrain = Mnist.builder().optUsage(Dataset.Usage.TRAIN).setSampling(	1024, true).build();
        mnistTrain.prepare();

        RandomAccessDataset mnistTest = Mnist.builder().optUsage(Dataset.Usage.TEST).setSampling(	1024, true).build();
        mnistTest.prepare();


        // Create a training configuration
        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                // Set the Max # of GPUs to 1
                .optDevices(Engine.getInstance().getDevices(1))
                // Set the Training Listener to log the training progress to the console *This is Default*
                .addTrainingListeners(TrainingListener.Defaults.logging("build/mlp"));

        // Create a trainer
        try (ai.djl.training.Trainer trainer = model.newTrainer(config)) {
            // Initialize the trainer with the input shape *The Size of the Image*
            Shape inputShape = new Shape(1, Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH);
            trainer.initialize(inputShape);
            // Train the model with the trainer, for 150 epochs, on the training dataset, and validate on the test dataset
            EasyTrain.fit(trainer, 150, mnistTrain, mnistTest);
            // Get the training result
            TrainingResult result = trainer.getTrainingResult();
            // Print the accuracy of the model
            System.out.println("Accuracy: " + result.getValidateEvaluation("Accuracy"));
            // Save the model
            Path modelDir = Paths.get("build/mlp");
            model.save(modelDir, "mlp");
        }
    }


}

