"""Default geometry: the user's garage, from the house blueprints.

Source: "Dogwood, Elevation A - 3 Car" (Aller Design Group / Greenland Homes, 2010),
sheet 2.1 MAIN FLOOR PLAN; ceiling from the plate-height callout.  Read from photos of
the paper plans on 2026-10-06 - verify the key numbers with a tape measure.

Plan dimensions are to exterior faces of framing ("all exterior frame dimensions include
1/2 in sheathing").  They are converted to the app's interior frame by assuming a
WALL = 0.4 ft (~4.75 in: 2x4 + sheathing + drywall) exterior wall.

    Plan reading (exterior, ft)                         Interior frame (ft)
    garage width  20'-0" + 12'-0" = 32'-0"              X 0 .. 31.2
    16070 OHGD (16'x7') centred in the 20' section      door 1  X 1.6 .. 17.6, Y 0
    8070 OHGD (8'x7') centred in the 12' section        door 2  X 21.6 .. 29.6, Y 3.0
    12' section front wall set back 3'-0"               front wall step at X 19.6
    12' section depth 22'-0" to its back wall           back wall Y 24.5 (X 19.6 .. 31.2)
    laundry/WIC wall 3'-3 1/2" forward of that          back wall Y 21.25 (X 0 .. 19.6)
    closet bump-out 2'-4 1/2" further, 4'-2" from the   bump X 4.17 .. 7.75 to Y 18.9
      left wall, 3'-7" wide
    plate height 9'-7 1/2" (garage slab ~1'-6" below    ceiling 9.625
      the house floor; entry steps in the left-rear corner are not modelled)

The table above is in PLAN orientation.  The house is built mirrored (see BUILT_MIRRORED), so
default_garage() flips X: in the app the single door is on the left and the double door on the
right.  Stalls are numbered as the user does: stall 1 = far side of the double door (right),
stall 2 = centre, stall 3 = single-door bay (left).

Radar: on the house wall over the centre stall (see the RADAR_* constants), facing the doors,
board upright (roll 0).
"""
from __future__ import annotations

import numpy as np

from .garage import Door, Garage
from .parking_zones import IgnoreZone, ParkingStall, default_car_zone
from .radar_pose import RadarPose
from .units import FT

WALL = 0.4  # assumed exterior wall thickness, ft

WIDTH = 32.0 - 2 * WALL  # 31.2
STEP_X = 20.0 - WALL  # 19.6: where the double-door section meets the single-door section
SETBACK = 3.0  # single-door wall, Y
BACK_RIGHT = 24.5  # back wall behind the single door
BACK_MAIN = 21.25  # house wall (laundry / WIC) behind the double door
BUMP_X0, BUMP_X1, BUMP_Y = 4.17, 7.75, 18.9  # closet bump-out
CEILING = 9.625

# floor-plan outline, counter-clockwise from the inside front-left corner (ft)
OUTLINE_FT = [
    (0.0, 0.0), (STEP_X, 0.0), (STEP_X, SETBACK), (WIDTH, SETBACK), (WIDTH, BACK_RIGHT),
    (STEP_X, BACK_RIGHT), (STEP_X, BACK_MAIN), (BUMP_X1, BACK_MAIN), (BUMP_X1, BUMP_Y),
    (BUMP_X0, BUMP_Y), (BUMP_X0, BACK_MAIN), (0.0, BACK_MAIN),
]


def _plan_garage() -> Garage:
    """The garage exactly as drawn on the plan (double door on the left, viewed from above)."""
    f = FT
    doors = [
        Door(width=16 * f, height=7 * f, x_left=(2.0 - WALL) * f, y=0.0),
        Door(width=8 * f, height=7 * f, x_left=(22.0 - WALL) * f, y=SETBACK * f),
    ]
    stalls = [
        # the two double-door stalls; the one nearest the side wall stops short of the closet bump-out
        ParkingStall("Stall", left=0.5 * f, right=9.4 * f, front=0.5 * f, rear=18.5 * f),
        ParkingStall("Stall", left=9.8 * f, right=19.2 * f, front=0.5 * f, rear=20.8 * f),
        ParkingStall("Stall", left=20.1 * f, right=30.7 * f, front=(SETBACK + 0.5) * f, rear=24.0 * f),
    ]
    return Garage(width=WIDTH * f, depth=BACK_RIGHT * f, ceiling_height=CEILING * f, doors=doors, stalls=stalls,
                  display_unit="ft", outline=[(x * f, y * f) for x, y in OUTLINE_FT])


def mirror_x(g: Garage) -> Garage:
    """Mirror a garage left-right (X -> width - X); stalls and doors are renumbered left to right."""
    W = g.width
    doors = sorted((Door(d.width, d.height, W - d.x_left - d.width, d.bottom, d.y) for d in g.doors),
                   key=lambda d: d.x_left)
    stalls = sorted((ParkingStall(s.name, W - s.right, W - s.left, s.front, s.rear,
                                  polygon=[(W - x, y) for x, y in reversed(s.polygon)]) for s in g.stalls),
                    key=lambda s: s.left)
    for i, s in enumerate(stalls):
        s.name = f"Stall {i + 1}"
    r = g.radar
    radar = RadarPose(W - r.x, r.y, r.z, (-r.azimuth_deg) % 360, r.elevation_deg, -r.roll_deg,
                      r.fov_azimuth_deg, r.fov_elevation_deg, r.max_range, r.flip_azimuth)
    return Garage(g.width, g.depth, g.ceiling_height, doors, stalls, radar, g.display_unit,
                  outline=[(W - x, y) for x, y in reversed(g.outline)],
                  ignore_zones=[IgnoreZone(z.name, [(W - x, y) for x, y in reversed(z.polygon)]) for z in g.ignore_zones])


# The house was built as the mirror image of the plan: seen from above with the doors at the
# bottom (as in the app), the single door is on the LEFT and the double door on the RIGHT.
# Confirmed by the user on 2026-10-08.
BUILT_MIRRORED = True

# Measured mount (user, 2026-10-08): ~9 ft above the floor on the house wall, facing the doors,
# "slightly left" of the centre stall's centre as seen standing at the doors looking in - that
# is the garage's -X side.  Tilt is fixed at 20 deg down by the 3D-printed enclosure.
RADAR_HEIGHT_FT = 9.0
RADAR_X_OFFSET_FT = -0.5  # from the centre of stall 2 (negative = toward stall 1); "slightly" ~ 6 in, unmeasured
RADAR_TILT_DEG = -20.0


def default_garage() -> Garage:
    f = FT
    g = _plan_garage()
    if BUILT_MIRRORED:
        g = mirror_x(g)
    # The user numbers stalls from the far side of the double door (the +X / right side when standing
    # at the doors looking in): stall 1 = far side of the double door, 2 = centre, 3 = single door.
    g.stalls = sorted(g.stalls, key=lambda s: -s.left)
    for i, s in enumerate(g.stalls):
        s.name = f"Stall {i + 1}"
    g.doors = sorted(g.doors, key=lambda d: -d.x_left)  # door 1 = double door, door 2 = single door
    centre = g.stalls[1]
    g.radar = RadarPose(
        x=(centre.left + centre.right) / 2 + RADAR_X_OFFSET_FT * f,
        y=(BACK_MAIN - 0.15) * f,  # board ~2 in off the house wall
        z=RADAR_HEIGHT_FT * f,
        azimuth_deg=180.0,  # facing the doors
        elevation_deg=RADAR_TILT_DEG,
        roll_deg=0.0,  # board upright, antenna block at the top
        max_range=15.0,  # v2 profile's CFAR range limit (cfarFovCfg)
        # Drive-in test 2026-10-08 (car into stall 1, the far side of the double door) confirmed TI's
        # standard azimuth sign for this board/firmware: +X is the radar's right.
        flip_azimuth=False,
    )
    _default_zones(g)
    return g


def _default_zones(g: Garage) -> None:
    """Car zones and clutter (ignore) zones as described by the user on 2026-10-08:
    clutter collects between the stalls on both sides of the centre stall, and at the HOUSE-WALL
    end of stall 3 (single door) and of stall 1 (far side of the double door).  Edit them on the
    map in Garage Geometry Mode."""
    f = FT
    s1, s2, s3 = g.stalls
    clutter_end = 3.0 * f  # house-wall end of stalls 1 and 3 kept out of their car zones
    s1.car_zone = default_car_zone(s1, rear_margin=clutter_end)
    s2.car_zone = default_car_zone(s2)
    s3.car_zone = default_car_zone(s3, rear_margin=clutter_end)
    z1, z2, z3 = (np.asarray(s.car_zone) for s in (s1, s2, s3))
    gap = 0.5 * f  # keep ignore zones this far from the car zones
    y_lo = 0.5 * f
    g.ignore_zones = [
        IgnoreZone("Between stalls 3 and 2", _rect(z3[:, 0].max() + gap, y_lo, z2[:, 0].min() - gap, s2.rear)),
        IgnoreZone("Between stalls 2 and 1", _rect(z2[:, 0].max() + gap, y_lo, z1[:, 0].min() - gap, s2.rear)),
        IgnoreZone("House end of stall 3", _rect(s3.left, z3[:, 1].max() + gap, s3.right, s3.rear)),
        IgnoreZone("House end of stall 1", _rect(s1.left, z1[:, 1].max() + gap, s1.right, s1.rear)),
    ]


def _rect(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
