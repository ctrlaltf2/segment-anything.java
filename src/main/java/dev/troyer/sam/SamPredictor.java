package dev.troyer.sam;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import javax.imageio.ImageIO;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;

public class SamPredictor {
    /**
     * Image that is currently selected
     */
    private SamImage image;

    /**
     * Image embedding (1x256x64x64)
     */
    // TODO: Couple SamImage, this embedding, and SamEncoder somehow
    private OnnxTensor imageEmbedding;

    /**
     * Encoder model handler
     */
    final private SamEncoder encoder;

    /**
     * Decoder model handler
     */
    final private SamDecoder decoder;

    /**
     * ONNX environment context
     */
    final private OrtEnvironment env;

    /**
     * Constructor
     */
    public SamPredictor(Path encoderModelPath, Path decoderModelPath) throws OrtException {
        env = OrtEnvironment.getEnvironment();

        encoder = new SamEncoder(env, encoderModelPath);
        decoder = new SamDecoder(env, decoderModelPath);

        // TODO: assert encoder.outputs.image_embeddings.shape == decoder.inputs.embeddings.shape
    }

    /**
     * Predict masks for given input prompts.
     * This is the full set of inputs to the decoder, rarely to be used directly
     *
     * @param pointCoords       Nx2 array of point prompts to the model. Each point is in pixel space.
     * @param pointLabels       Nx1 array of labels for each point in pointCoords. 1 indicates foreground,
     *                          0 indicates background.
     * @param maskInput         1x1x256x256 Low res mask input to the model, usually from a previous prediciton iteration.
     * @param hasMaskInput      true if maskInput is to be used
     * @param originalImageSize original image dimensions
     * @return TBD
     */
    public void predict(float[][] pointCoords, float[] pointLabels, OnnxTensor maskInput, boolean hasMaskInput, int[] originalImageSize) throws OrtException {
        // TODO: Move this to decoder
        // Prep native parameters to be parameters to the model
        LinkedHashMap<String, OnnxTensor> inputs = new LinkedHashMap<>();
        inputs.put("image_embeddings", this.imageEmbedding);
        inputs.put("point_coords", OnnxTensor.createTensor(env, pointCoords));
        inputs.put("point_labels", OnnxTensor.createTensor(env, pointLabels));
        inputs.put("mask_input", OnnxTensor.createTensor(env, maskInput));
        inputs.put("has_mask_input", OnnxTensor.createTensor(env, new float[]{hasMaskInput ? 1.0f : 0.0f}));
        inputs.put("orig_im_size", OnnxTensor.createTensor(env, originalImageSize));

        // this.model.decoder.run(inputs)
    }

    /**
     * Main interface, probably the most common use case
     *
     * @param constraints Queries/constraints to the model, given current image
     * @return TBD
     */
    public void predict(SamConstraint[] constraints) throws OrtException {
        assert image != null;

        // Unpack constraints into form usable by the model
        float[][] pointCoords = new float[constraints.length][2];
        float[] pointLabels = new float[constraints.length];

        for (int iConstraint = 0; iConstraint < constraints.length; ++iConstraint) {
            SamConstraint thisConstraint = constraints[iConstraint];

            // TODO: confirm ordering
            pointCoords[iConstraint][0] = (float) thisConstraint.x;
            pointCoords[iConstraint][1] = (float) thisConstraint.y;

            pointLabels[iConstraint] = (float) thisConstraint.type.ordinal();
        }

        // TODO: confirm ordering of h/w
        final int[] originalImageSize = new int[]{image.originalWidth, image.originalHeight};

        // ONNX model at the moment always expects a mask input.
        // I think it's because ONNX models' parameters might be fixed, meaning optional
        // items must still be populated (and tagged separately with a bool).
        // items don't need explicitly set https://docs.oracle.com/javase/specs/jls/se17/html/jls-4.html#jls-4.12.5

        final long[] maskInputShape = this.decoder.getMaskInputsShape();
        final OnnxTensor maskInput = OnnxTensor.createTensor(
                env,
                new float[(int) maskInputShape[0]][(int) maskInputShape[1]][(int) maskInputShape[2]][(int) maskInputShape[3]]
        );

        // TODO: return
        predict(pointCoords, pointLabels, maskInput, false, originalImageSize);
    }

    /**
     * Set current image by file path
     *
     * @param imagePath Path to the image. Isn't validated or anything, assumes it's an image
     */
    public void setImageFromPath(Path imagePath) throws IOException, OrtException {
        // TODO: Error handling
        image = new SamImage(ImageIO.read(new File(imagePath.toString())));
        onImageUpdate();
    }

    /**
     * Should run when this->image updates. Re-encodes the image for efficient prompting later.
     */
    private void onImageUpdate() throws OrtException {
        var forwardResult = encoder.forward(env, image);
        assert forwardResult.get("image_embeddings").isPresent();
        imageEmbedding = (OnnxTensor) forwardResult.get("image_embeddings").get();
    }
}
