package dev.troyer.sam;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import javax.imageio.ImageIO;
import java.util.Arrays;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.enums.Mode;
import org.nd4j.enums.ImageResizeMethod;


class SamImage {
    /**
     * Underlying data for the image, post-transform
     * Shape (num in batch == 1, height == 1024, width == 1024, num channels == 3)
     */
    private final double[][][][] data;

    /**
     * Original height, px
     */
    public final int original_height;

    /**
     * Original width, px
     */
    public final int original_width;

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
        // TODO: support 16-bit images?
        int[] pixels = img.getRGB(0, 0, img.getWidth(), img.getHeight(), null, 0, img.getWidth());

        // (batch size == 1, height, width, channels == 3)
        INDArray img_tensor = Nd4j.create(DataType.DOUBLE, 1, img.getHeight(), img.getWidth(), 3);

        for (int i = 0; i < pixels.length; i++) {
            int x = i % img.getWidth();
            int y = i / img.getWidth();

            // unpack pixel into RGB values
            final double r = ((pixels[i] >> 16) & 0xFF);
            final double g = ((pixels[i] >> 8) & 0xFF);
            final double b = (pixels[i] & 0xFF);

            img_tensor.putScalar(0, y, x, 0, r);
            img_tensor.putScalar(0, y, x, 1, g);
            img_tensor.putScalar(0, y, x, 2, b);
        }

        // Standardize the image (Z-score transform)
        INDArray mean = img_tensor.mean(0, 1, 2)
                .reshape(1, 1, 1, 3)
                .broadcast(1, img.getHeight(), img.getWidth(), 3);

        INDArray std = img_tensor.std(0, 1, 2)
                .reshape(1, 1, 1, 3)
                .broadcast(1, img.getHeight(), img.getWidth(), 3);

        img_tensor.subi(mean).divi(std);

        // Resize proportionally such that longest side is 1024 pixels
        final double scale = 1024.0 / Math.max(img.getWidth(), img.getHeight());
        final int new_width  = Math.min((int) ( img.getWidth() * scale + 0.5), 1024);
        final int new_height = Math.min((int) (img.getHeight() * scale + 0.5), 1024);

        // why are there literally 5 different ways to resize an image with this library and why are 4 of them wrong
        INDArray new_size = Nd4j.createFromArray(new int[]{new_height, new_width});
        INDArray resized = Nd4j.image.imageResize(img_tensor, new_size, ImageResizeMethod.ResizeBilinear);

        // Pad such that the image is 1024x1024
        final int pad_width = Math.max(1024 - new_width, 0);
        final int pad_height = Math.max(1024 - new_height, 0);

        // docs were not clear on this __at all__, but it's the start/end padding for each axis, in order of dimension.
        INDArray padding = Nd4j.createFromArray(new int[][]{
                {0, 0}, // batch padding
                {0, pad_height},
                {0, pad_width},
                {0, 0} // pad channel
        });

        INDArray padded = Nd4j.image.pad(resized, padding, Mode.CONSTANT, 0.0);

        assert padded.shape()[0] == 1;
        assert padded.shape()[1] == 1024;
        assert padded.shape()[2] == 1024;
        assert padded.shape()[3] == 3;

        // and into a double array you go (cursed way because NDArray cannot export 4D matrices apparently)
        data = new double[1][1024][1024][3];

        for (int i = 0; i < 1024; i++)
            for (int j = 0; j < 1024; j++)
                for (int k = 0; k < 3; k++)
                    this.data[0][i][j][k] = padded.getDouble(0, i, j, k);

        this.original_height = img.getHeight();
        this.original_width = img.getWidth();
    }

    /**
     * Get the data for the image, post transfom and ready for encoding
     */
    public double[][][][] post_transform() {
        return this.data;
    }

    private void dump(INDArray imageTensor, String filePath) {
        int height = (int) imageTensor.shape()[1];
        int width = (int) imageTensor.shape()[2];

        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int r = (int) (imageTensor.getDouble(0, y, x, 0));
                int g = (int) (imageTensor.getDouble(0, y, x, 1));
                int b = (int) (imageTensor.getDouble(0, y, x, 2));

                r = Math.max(0, Math.min(r, 255));
                g = Math.max(0, Math.min(g, 255));
                b = Math.max(0, Math.min(b, 255));

                int rgb = (r << 16) | (g << 8) | b;
                image.setRGB(x, y, rgb);
            }
        }

        try {
            String format = filePath.substring(filePath.lastIndexOf(".") + 1);
            ImageIO.write(image, format, new File(filePath));
        } catch (IOException e) {
            System.err.println("Error: Unable to save the image.");
            e.printStackTrace();
        }
    }
}
