package dev.troyer.sam;

import ai.onnxruntime.OrtException;
import dev.troyer.sam.SamImage;
import dev.troyer.sam.SamEncoder;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.ops.NDImage;
import org.nd4j.linalg.factory.Nd4j;

public class Main {
    public static void main(String[] args) throws OrtException {
        String imagePath = "/home/caleb/Pictures/example.jpg";
        BufferedImage image = null;

        try {
            image = ImageIO.read(new File(imagePath));
        } catch (IOException e) {
            System.err.println("Error while loading image: " + e.getMessage());
            return;
        }

        SamImage samImage = new SamImage(image);

        try {
            Path model_path = Path.of("./src/main/resources/data/vit_b/encoder-vit_b.quant.onnx");
            SamEncoder encoder = new SamEncoder(model_path);
            var out = encoder.forward(samImage);
            System.out.println(out);
        } catch (OrtException e) {
            System.err.println("Error while loading model: " + e.getMessage());
            return;
        }
    }
}
