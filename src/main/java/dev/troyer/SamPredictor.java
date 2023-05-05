package dev.troyer;

import dev.troyer.SamDecoder;
import dev.troyer.SamEncoder;

import ai.onnxruntime.OrtSession;

public class SamPredictor {
    // encoder model
    private SamEncoder encoder;

    // decoder model
    private SamDecoder decoder;

    // Constructor
    public SamPredictor() {

    }
}
