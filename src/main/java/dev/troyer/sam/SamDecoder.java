package dev.troyer.sam;

import ai.onnxruntime.*;

import java.nio.file.Path;
import java.util.Map;

public class SamDecoder {
    /**
     * Loaded ONNX model
     */
    private final OrtSession model;

    /**
     * @param env       ONNX environment context
     * @param modelPath Path to the model file
     */
    public SamDecoder(OrtEnvironment env, Path modelPath) throws OrtException {
        this.model = env.createSession(modelPath.toString());
        // TODO: Verify input/output names. These could vary based on how/who exported the model.
    }

    /**
     * @return shape of the mask inputs parameter
     */
    public long[] getMaskInputsShape() {
        // TODO: better error handling, don't assume output with this name exists
        final Map<String, NodeInfo> inputsInfo;
        try {
            inputsInfo = model.getInputInfo();
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }

        final NodeInfo maskInputNodeInfo = inputsInfo.get("mask_input");
        final TensorInfo maskInputTensorInfo = (TensorInfo) maskInputNodeInfo.getInfo();
        return maskInputTensorInfo.getShape();
    }
}
