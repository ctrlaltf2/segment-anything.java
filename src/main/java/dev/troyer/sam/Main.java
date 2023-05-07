package dev.troyer.sam;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;

public class Main {
    public static void main(String[] args) {
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        System.out.println("Probably setup env");
    }
}