package dev.troyer.sam;

import ai.onnxruntime.OrtException;

import java.io.IOException;
import java.nio.file.Path;

public class Main {
    public static void main(String[] args) throws IOException, OrtException {
        SamPredictor predictor = new SamPredictor(
                Path.of("./src/main/resources/data/vit_b/encoder-vit_b.quant.onnx"),
                Path.of("./src/main/resources/data/vit_b/decoder-vit_b.quant.onnx")
        );

        predictor.setImageFromPath(
                Path.of("./test/assets/mingjun-liu-mVWqCdTHfxs-unsplash.jpg")
        );

        final SamResult result = predictor.predict(new SamConstraint[]{
                new SamConstraint(1980, 1200, ConstraintType.FOREGROUND)
        });
    }
}
