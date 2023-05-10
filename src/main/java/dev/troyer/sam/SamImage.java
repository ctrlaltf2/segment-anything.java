package dev.troyer.sam;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import javax.imageio.ImageIO;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.ops.NDImage;
import org.nd4j.linalg.factory.Nd4j;

class SamImage {
    /**
     * Underlying data for the image, post-transform
     */
    private final INDArray data;

    /*
    /**
     * Image from Path
     *
     * @param img_path Path to the image file
    public SamImage(Path img_path) {
        // Load into a buffered image

        BufferedImage image = null;
        try {
            // Read the image file from disk into a BufferedImage
            image = ImageIO.read(img_path.toFile());
        } catch (IOException e) {
            System.out.println("Error reading image file: " + e.getMessage());
            throw e;
        }

        this(image);
    }
    */

    /**
     * Image from BufferedImage
     */
    public SamImage(BufferedImage img) {

        // To NDArray
        int[] pixels = img.getRGB(0, 0, img.getWidth(), img.getHeight(), null, 0, img.getWidth());

        float[][] float_pixels = new float[pixels.length];

        for(int i = 0; i < pixels.length; i++)

        // Process using NDImage
        NDImage image_processor = new NDImage();
    }

    public INDArray asNDArray() {
        return this.data;
    }
}
