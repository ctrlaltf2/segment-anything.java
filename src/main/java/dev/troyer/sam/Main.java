package dev.troyer.sam;

import java.io.IOException;
import java.nio.file.Path;

public class Main {
    public static void main(String[] args) throws IOException {
        Path imagePath = Path.of("./test/assets/mingjun-liu-mVWqCdTHfxs-unsplash.jpg");
        SamPredictor predictor = new SamPredictor();
        // Predictor is sort manually driven for now, better design comes later after I know fully what this looks like
        predictor.setImageFromPath(imagePath);
    }
}
