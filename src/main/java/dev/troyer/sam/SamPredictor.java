package dev.troyer.sam;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import javax.imageio.ImageIO;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;

public class SamPredictor {
    /** Image that is currently selected */
    private SamImage image;

    /** Image embedding (1x256x64x64) */
    // TODO: Couple SamImage, this embedding, and SamEncoder somehow
    private float[][][][] imageEmbedding;

    /** Encoder model handler */
    private SamEncoder encoder;

    /** Decoder model handler */
    private SamDecoder decoder;

    /** ONNX environment context*/
    private OrtEnvironment env;

    /** Constructor */
    public SamPredictor() {
        env = OrtEnvironment.getEnvironment();

        // Load encoder model
        // TODO: selector or something, less hard-codey
        Path encoderModelPath = Path.of("./src/main/resources/data/vit_b/encoder-vit_b.quant.onnx");
        try {
            encoder = new SamEncoder(env, encoderModelPath);
        } catch (OrtException e) {
            // TODO: better error handling etc. (aka not just doing this)
            throw new RuntimeException(e);
        }

        // Load decoder model
    }

    /**
     * Predict masks for given input prompts.
     *
     * @param pointCoords Nx2 array of point prompts to the model. Each point is in pixel space.
     * @param pointLabels Nx1 array of labels for each point in pointCoords. 1 indicates foreground,
     *                    0 indicates background.
     *
     * @return sdf
     */
    /*
    public OnnxTensor predict(float[][] pointCoords, float[][] pointLabels) {

    }
    */

    /**
     * Set current image by file path
     * @param imagePath Path to the image. Isn't validated or anything, assumes it's an image
     */
    public void setImageFromPath(Path imagePath) throws IOException {
        // TODO: Error handling
        image = new SamImage(ImageIO.read(new File(imagePath.toString())));
        onImageUpdate();
    }

    /**
     * Should run when this->image updates. Re-encodes the image for efficient prompting later.
     */
    private void onImageUpdate() {
        try {
            // TODO: Error handling
            var result = encoder.forward(env, image).get(0);
            imageEmbedding = (float[][][][]) result.getValue(); // 1x256x64x64
            System.out.println(result);
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
    }
}
