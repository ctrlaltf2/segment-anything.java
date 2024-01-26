package dev.troyer.sam;

/**
 * Represents a result of a query to the segment-anything model as a whole
 */
public class SamResult {
    /**
     * Mask(s) returned by the model
     */
    final public float[][][][] masks;

    /**
     * iou_predictions
     */
    final public float[][] iouPredictions;

    /**
     * low_res_masks
     */
    final public float[][][][] lowResMasks;

    /**
     * Constructor, initializes members
     * @param masks Mask(s) returned by the model
     * @param iouPredictions iou_predictions
     * @param lowResMasks low_res_masks
     */
    public SamResult(float[][][][] masks, float[][] iouPredictions, float[][][][] lowResMasks) {
        this.masks = masks;
        this.iouPredictions = iouPredictions;
        this.lowResMasks = lowResMasks;
    }
}
