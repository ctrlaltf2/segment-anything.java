package dev.troyer.sam;

import java.nio.file.Path;
import java.util.LinkedHashMap;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

class SamEncoder {
    private final OrtSession model;

    private final int imageSize;

    /**
     * @param modelPath Path to the model file
     * @param imageSize Size of the image the encoder model was trained on (usually 1024)
     */
    public SamEncoder(OrtEnvironment env, Path modelPath, int imageSize) throws OrtException {
        this.imageSize = imageSize;
        this.model = env.createSession(modelPath.toString());
    }

    /**
     * @param modelPath Path to the model file
     */
    public SamEncoder(OrtEnvironment env, Path modelPath) throws OrtException {
        this(env, modelPath, 1024);
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
