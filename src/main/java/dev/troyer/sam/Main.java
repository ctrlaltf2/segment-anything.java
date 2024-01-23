package dev.troyer.sam;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import javax.imageio.ImageIO;
import java.nio.FloatBuffer;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;

public class Main {
    public static void main(String[] args) throws OrtException {
        Path imagePath = Path.of("./test/assets/mingjun-liu-mVWqCdTHfxs-unsplash.jpg");
        BufferedImage image = null;

        try {
            image = ImageIO.read(new File(imagePath.toString()));
        } catch (IOException e) {
            System.err.println("Error while loading image: " + e.getMessage());
            return;
        }

        OrtEnvironment env = OrtEnvironment.getEnvironment();
        SamImage samImage = new SamImage(image);

        try {
            Path modelPath = Path.of("./src/main/resources/data/vit_b/encoder-vit_b.quant.onnx");
            SamEncoder encoder = new SamEncoder(env, modelPath);
            var out = encoder.forward(env, samImage);
            System.out.println(out);
            var result = out.get(0);
            float[][][][] value = (float[][][][]) result.getValue(); // 1x256x64x64
            var placeholder = out.get(0);
        } catch (OrtException e) {
            System.err.println("Error while loading model: " + e.getMessage());
            return;
        }
    }
}
