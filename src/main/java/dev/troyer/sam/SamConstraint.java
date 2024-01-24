package dev.troyer.sam;

/**
 * Represents a constraint to the SAM decoder model.
 * A constraint is the foreground/background parts you select in
 * all those nice Segment Anything frontends.
 */
public class SamConstraint {
    /**
     * Column in image (pixel space). Origin is top left.
     */
    final public int x;

    /**
     * Row in image (pixel space). Origin is top left.
     */
    final public int y;

    /**
     * Type of constraint (foreground/background)
     */
    final public ConstraintType type;

    public SamConstraint(int x, int y, ConstraintType type) {
        this.x = x;
        this.y = y;
        this.type = type;
    }
}
