package dev.troyer.sam;

import dev.troyer.sam.SamDecoder;
import dev.troyer.sam.SamEncoder;

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
