package dev.troyer.sam;

import java.nio.file.Path;
import java.util.Map;
import java.util.LinkedHashMap;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OnnxTensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.onnxruntime.runner.OnnxRuntimeRunner;

import dev.troyer.sam.SamImage;

class SamEncoder {
    private final OnnxRuntimeRunner model;

    private final int image_size;

    /**
     * @param model_path Path to the model file
     * @param image_size Size of the image the encoder model was trained on (usually 1024)
     */
    public SamEncoder(Path model_path, int image_size) throws OrtException {
        this.image_size = image_size;
        this.model = OnnxRuntimeRunner.builder()
                .modelUri(model_path.toString())
                .build();
    }

    /**
     * @param model_path Path to the model file
     */
    public SamEncoder(Path model_path) throws OrtException {
        this(model_path, 1024);
    }

    /**
     * @param input Image tensor to encode
     * @return Embedding for the image tensor
     */
    public Map<String,INDArray> forward(SamImage input) throws OrtException {
        LinkedHashMap<String, INDArray> inputs = new LinkedHashMap<>();
        inputs.put("x", input.asNDArray());
        return model.exec(inputs);
    }
}
