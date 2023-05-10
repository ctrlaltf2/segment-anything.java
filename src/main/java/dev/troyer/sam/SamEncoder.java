package dev.troyer.sam;

import java.nio.file.Path;
import java.util.LinkedHashMap;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

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
    public OrtSession.Result forward(OrtEnvironment env, SamImage input) throws OrtException {
        LinkedHashMap<String, OnnxTensor> inputs = new LinkedHashMap<>();
        inputs.put("x", input.asTensor(env));
        return model.run(inputs);
    }
}
