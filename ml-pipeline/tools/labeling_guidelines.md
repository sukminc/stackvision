
# Labeling Guidelines — StackVision (Venue-first)

Goal: Instance segmentation of poker chips by denomination for ONE venue preset.

## Classes (edit if needed)
- "100", "500", "1000", "5000", "25000"

## Do label
- Each visible chip (even partial) with a polygon/mask (preferred) or bbox (acceptable for v1).
- Mixed stacks and spread chips.
- Only chips whose denomination you are confident about.

## Do NOT label
- Blurry/unreadable chips.
- Background graphics, cards, phones, chip racks without visible denominations.

## Splits
- 80/20 train/val split **by scene**: avoid near-duplicate frames in both sets.

## Image variety
- Angles 10–25° top-down, a few side-ish.
- Different lighting/felts.
- Mix single-color stacks and messy stacks.

## File layout (YOLOv8-seg)
datasets/pokerplace_v1/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt  # YOLO seg format
  labels/val/*.txt

## Quality tips
- Keep short side 640–960px.
- Avoid extreme motion blur.
- Ensure at least 250 instances per class across the dataset if possible.
