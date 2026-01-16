# Shadow-Files

Mini app: realistic shadow generator (Python CLI).

## Setup

Install deps:

- Python 3.9+
- `pip install -r requirements.txt`

## Usage

Basic (auto cutout using GrabCut if no alpha/mask):

```
python shadow_generator.py \
	--foreground "./25_1107O_11974 PB + 1 - Photo Calendar B_Lamborghini HAS.JPG" \
	--background "./B_Lamborghini Red.JPG" \
	--angle 135 --elevation 35
```

Optional mask and depth map:

```
python shadow_generator.py \
	--foreground "./subject.png" \
	--background "./background.jpg" \
	--mask "./subject_mask.png" \
	--depth "./depth.png" \
	--angle 120 --elevation 40 --opacity 0.65
```

Outputs (in `outputs/` by default):

- `composite.png`
- `shadow_only.png`
- `mask_debug.png`

## Notes

- Light angle is the **direction of light** in screen space (0° = right, 90° = down). Shadow is cast opposite.
- Elevation controls shadow length: lower elevation → longer shadow.
- If the foreground has an alpha channel, it’s used as the cutout.
- If no alpha or mask is provided, the script uses GrabCut with a centered rectangle.
# Shadow-Files