package org.example;

import ai.djl.MalformedModelException;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.nn.Block;
import ai.djl.Model;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * This class is used to predict the class of an image using a trained MLP model.
 */
public class ImageDetect {

    private static final Logger logger = LoggerFactory.getLogger(ImageDetect.class);

    public static void main(String[] args) throws TranslateException, MalformedModelException, IOException {

        Classifications classifications =  predictImage("src/main/resources/9.png");

        // Print out the results
        logger.info("{}", classifications);

    }


    /**
     * This method is used to predict the class of an image.
     * It first loads the image and the trained MLP model.
     * Then, it creates a translator and a predictor.
     * Finally, it uses the predictor to predict the classes of the image.
     *
     * @param imagePath The path of the image to predict.
     * @return The predicted classes of the image.
     */
    public static Classifications predictImage(String imagePath) throws IOException, TranslateException, MalformedModelException {
        // Load the image
        Path imageFile = Paths.get(imagePath);
        Image img = ImageFactory.getInstance().fromFile(imageFile);
        // Create a Neural Network
        Block block =
                new Mlp(
                        // Define the input size of the neural network (28x28 images) = 784 pixels
                        Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH,
                        // Define the number of classes in the dataset (10 classes) = 0-9
                        Mnist.NUM_CLASSES,
                        // Define the number of neurons in each layer of the neural network
                        // 128 in the first hidden layer and 64 in the second hidden layer
                        new int[]{128, 64});
        //Load the model
        String modelName = "mlp";
        Model model = Model.newInstance(modelName);
        model.setBlock(block);
        Path modelDir = Paths.get("build/mlp");
        model.load(modelDir);


        // Create a list of classes for the translator to use
        List<String> classes =
                IntStream.range(0, 10).mapToObj(String::valueOf).collect(Collectors.toList());

        // Create a translator
        Translator<Image, Classifications> translator =
                ImageClassificationTranslator.builder()
                        // Convert the image to a tensor
                        .addTransform(new ToTensor())
                        // set the list of classes
                        .optSynset(classes)
                        // convert the output to a softmax probability distribution
                        .optApplySoftmax(true)
                        .build();
        // Create a predictor
        try (Predictor<Image, Classifications> predictor = model.newPredictor(translator)){
            // Run the prediction
            return predictor.predict(img);
        }
    };



}
