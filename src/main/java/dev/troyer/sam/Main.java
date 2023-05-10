package dev.troyer.sam;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import javax.imageio.ImageIO;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;

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

        OrtEnvironment env = OrtEnvironment.getEnvironment();
        SamImage samImage = new SamImage(image);

        try {
            Path model_path = Path.of("./src/main/resources/data/vit_b/encoder-vit_b.quant.onnx");
            SamEncoder encoder = new SamEncoder(env, model_path);
            var out = encoder.forward(env, samImage);
            System.out.println(out);
        } catch (OrtException e) {
            System.err.println("Error while loading model: " + e.getMessage());
            return;
        }
    }
}
