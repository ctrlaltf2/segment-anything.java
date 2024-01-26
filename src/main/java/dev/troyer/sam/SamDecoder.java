package dev.troyer.sam;

import ai.onnxruntime.*;

import java.nio.file.Path;
import java.util.LinkedHashMap;
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
    public long[] getMaskInputsShape() throws OrtException {
        // TODO: better error handling, don't assume output with this name exists
        final Map<String, NodeInfo> inputsInfo = model.getInputInfo();
        final NodeInfo maskInputNodeInfo = inputsInfo.get("mask_input");
        final TensorInfo maskInputTensorInfo = (TensorInfo) maskInputNodeInfo.getInfo();
        return maskInputTensorInfo.getShape();
    }

    public SamResult forward(OrtEnvironment env, OnnxTensor imageEmbedding,
                                     float[][][] pointCoords, float[][] pointLabels,
                                     OnnxTensor maskInput, boolean hasMaskInput,
                                     int[] originalImageSize) throws OrtException {
        LinkedHashMap<String, OnnxTensor> inputs = new LinkedHashMap<>();
        inputs.put("image_embeddings", imageEmbedding);
        inputs.put("point_coords", OnnxTensor.createTensor(env, pointCoords));
        inputs.put("point_labels", OnnxTensor.createTensor(env, pointLabels));
        inputs.put("mask_input", maskInput);
        inputs.put("has_mask_input", OnnxTensor.createTensor(env, new float[]{hasMaskInput ? 1.0f : 0.0f}));
        inputs.put("orig_im_size", OnnxTensor.createTensor(env, new float[]{originalImageSize[0], originalImageSize[1]}));

        OrtSession.Result result = model.run(inputs);

        assert result.get("masks").isPresent();
        assert result.get("iou_predictions").isPresent();
        assert result.get("low_res_masks").isPresent();

        // TODO: use cursed onnx api to verify OnnxValue rank instead of assuming
        return new SamResult(
                (float[][][][]) result.get("masks").get().getValue(),
                (float[][]) result.get("iou_predictions").get().getValue(),
                (float[][][][]) result.get("low_res_masks").get().getValue()
        );
    }
}
