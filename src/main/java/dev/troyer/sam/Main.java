package dev.troyer.sam;

import dev.troyer.sam.SamImage;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.ops.NDImage;
import org.nd4j.linalg.factory.Nd4j;

public class Main {
    public static void main(String[] args) {
        String imagePath = "/home/caleb/Pictures/example.jpg";
        BufferedImage image = null;

        try {
            image = ImageIO.read(new File(imagePath));
        } catch (IOException e) {
            System.err.println("Error while loading image: " + e.getMessage());
            return;
        }
        
        SamImage samImage = new SamImage(image);
    }
}
