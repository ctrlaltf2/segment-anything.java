package dev.troyer.sam;

import java.nio.file.Path;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.*;

class SamEncoder {
    /**
     * Loaded ONNX model
     */
    private final OrtSession model;

    private final int imageSize;

    /**
     * @param env       ONNX environment context
     * @param modelPath Path to the model file
     */
    public SamEncoder(OrtEnvironment env, Path modelPath) throws OrtException {
        this.model = env.createSession(modelPath.toString());

        // Verify inputs/outputs
        // TODO: Exception instead of this
        assert this.model.getInputNames().equals(
                new HashSet<>(List.of("x"))
        );

        assert this.model.getInputNames().equals(
                new HashSet<>(List.of("image_embeddings"))
        );

        Map<String, NodeInfo> inputsInfo = model.getInputInfo();
        final NodeInfo imageNodeInfo = inputsInfo.get("x");
        final TensorInfo imageTensorInfo = (TensorInfo) imageNodeInfo.getInfo();
        this.imageSize = (int) imageTensorInfo.getShape()[2];
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
