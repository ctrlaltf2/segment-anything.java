package dev.troyer.sam;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Path;
import java.util.HashMap;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OnnxTensor;

class SamEncoder {
    private final OrtSession model;

    private final int image_size;

    /**
     * @param model_path Path to the model file
     * @param image_size Size of the image the encoder model was trained on (usually 1024)
     */
    public SamEncoder(OrtEnvironment env, Path model_path, int image_size) throws OrtException {
        this.image_size = image_size;
        this.model = env.createSession(model_path.toString());
    }

    /**
     * @param model_path Path to the model file
     */
    public SamEncoder(OrtEnvironment env, Path model_path) throws OrtException {
        this(env, model_path, 1024);
    }

    /**
     * @param input Image tensor to encode
     * @return Embedding for the image tensor
     */
    public OrtSession.Result forward(OnnxTensor input) throws OrtException {
        HashMap<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("x", input);
        return model.run(inputs);
    }

    public forward(BufferedImage img) {
    }

    public forward(Path p) {

    }
}
