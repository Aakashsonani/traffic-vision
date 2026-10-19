"""Stable semantic IDs; never infer a sign meaning from a substring at runtime."""

# raw model label -> (canonical ID, display name, category, plain meaning)
SIGN_DEFINITIONS = {
    "SpeedLimit20": ("speed_20", "Speed limit 20 km/h", "Regulatory", "Maximum posted speed: 20 km/h."),
    "SpeedLimit30": ("speed_30", "Speed limit 30 km/h", "Regulatory", "Maximum posted speed: 30 km/h."),
    "SpeedLimit50": ("speed_50", "Speed limit 50 km/h", "Regulatory", "Maximum posted speed: 50 km/h."),
    "SpeedLimit70": ("speed_70", "Speed limit 70 km/h", "Regulatory", "Maximum posted speed: 70 km/h."),
    "SpeedLimit80": ("speed_80", "Speed limit 80 km/h", "Regulatory", "Maximum posted speed: 80 km/h."),
    "SpeedLimit100": ("speed_100", "Speed limit 100 km/h", "Regulatory", "Maximum posted speed: 100 km/h."),
    "SpeedLimit120": ("speed_120", "Speed limit 120 km/h", "Regulatory", "Maximum posted speed: 120 km/h."),
    "EndSpeedLimit80": ("end_speed_80", "End of 80 km/h limit", "Regulatory", "End of the posted 80 km/h speed restriction."),
    "IceSnow": ("ice_snow", "Ice or snow", "Warning", "Warning of ice or snow on the road."),
    "EndAllRestrictions": ("end_restrictions", "End of restrictions", "Regulatory", "End of the indicated local restrictions."),
    "TurnRightAhead": ("turn_right", "Turn right ahead", "Direction", "Mandatory right turn ahead."),
    "TurnLeftAhead": ("turn_left", "Turn left ahead", "Direction", "Mandatory left turn ahead."),
    "AheadOnly": ("straight", "Straight ahead", "Direction", "Continue straight ahead only."),
    "KeepLeft": ("keep_left", "Keep left", "Direction", "Pass the obstruction on its left."),
    "KeepRight": ("keep_right", "Keep right", "Direction", "Pass the obstruction on its right."),
    "GoStraightOrLeft": ("straight_left", "Straight or left", "Direction", "Permitted directions shown: straight ahead or left."),
    "GoStraightOrRight": ("straight_right", "Straight or right", "Direction", "Permitted directions shown: straight ahead or right."),
    "RoundaboutMandatory": ("roundabout", "Roundabout", "Direction", "Follow the indicated roundabout direction."),
    "Stop": ("stop", "Stop", "Regulatory", "Stop sign."),
    "Yield": ("yield", "Give way", "Regulatory", "Give way to priority traffic."),
    "NoEntry": ("no_entry", "No entry", "Prohibition", "Vehicle entry is prohibited from this direction."),
    "NoVehicles": ("no_vehicles", "No vehicles", "Prohibition", "Vehicles prohibited."),
    "VehiclesOver3.5TonsProhibited": ("no_heavy_vehicles", "No vehicles over 3.5 t", "Prohibition", "Vehicles over the indicated weight prohibited."),
    "NoPassing": ("no_passing", "No overtaking", "Prohibition", "Overtaking prohibited."),
    "NoPassingTrucks": ("no_passing_trucks", "No overtaking by trucks", "Prohibition", "Overtaking by trucks prohibited."),
    "EndNoPassing": ("end_no_passing", "End of no overtaking", "Regulatory", "End of the overtaking restriction."),
    "EndNoPassingByTrucks": ("end_no_passing_trucks", "End of truck overtaking restriction", "Regulatory", "End of the truck overtaking restriction."),
    "SpeedLimit60": ("speed_60", "Speed limit 60 km/h", "Regulatory", "Maximum posted speed: 60 km/h."),
    "RightOfWayCrossing": ("priority_crossing", "Priority at next junction", "Regulatory", "Priority at the next junction."),
    "PriorityRoad": ("priority_road", "Priority road", "Regulatory", "Priority road designation."),
    "DangerousCurveLeft": ("curve_left", "Bend to the left", "Warning", "Warning of a left bend ahead."),
    "DangerousCurveRight": ("curve_right", "Bend to the right", "Warning", "Warning of a right bend ahead."),
    "DoubleCurve": ("double_curve", "Double bend", "Warning", "Warning of successive bends."),
    "GeneralCaution": ("caution", "General caution", "Warning", "General hazard warning."),
    "BumpyRoad": ("bumpy_road", "Uneven road", "Warning", "Warning of an uneven road surface."),
    "SlipperyRoad": ("slippery", "Slippery road", "Warning", "Warning of a slippery road surface."),
    "RoadNarrowRight": ("narrows_right", "Road narrows on right", "Warning", "Road narrows on the right ahead."),
    "RoadWork": ("road_work", "Road works", "Warning", "Road works ahead."),
    "TrafficSignals": ("signals_ahead", "Traffic signals ahead", "Warning", "Warning sign for traffic signals ahead; not a live signal state."),
    "Pedestrians": ("pedestrians", "Pedestrian crossing", "Warning", "Warning of pedestrians or a pedestrian crossing."),
    "ChildrenCrossing": ("children", "Children crossing", "Warning", "Warning of children crossing."),
    "BicyclesCrossing": ("bicycles", "Bicycle crossing", "Warning", "Warning of cyclists crossing."),
    "WildAnimals": ("animals", "Wild animals", "Warning", "Warning of wild animals crossing."),
    "do_not_turn_l": ("no_left", "No left turn", "Prohibition", "Left turns prohibited."),
    "do_not_turn_r": ("no_right", "No right turn", "Prohibition", "Right turns prohibited."),
    "do_not_u_turn": ("no_u_turn", "No U-turn", "Prohibition", "U-turns prohibited."),
    "u_turn": ("u_turn", "U-turn", "Direction", "Sign indicates a U-turn."),
    "enter_left_lane": ("enter_left_lane", "Enter left lane", "Direction", "Sign indicates entry into the left lane."),
    "left_right_lane": ("left_right", "Left or right", "Direction", "Sign indicates left and right directions."),
    "no_parking": ("no_parking", "No parking", "Prohibition", "Parking prohibited."),
    "do_not_stop": ("no_stopping", "No stopping", "Prohibition", "Stopping prohibited."),
    "parking": ("parking", "Parking", "Information", "Designated parking sign."),
    "bus_stop": ("bus_stop", "Bus stop", "Information", "Designated bus stop."),
    "railway_crossing": ("railway", "Railway crossing", "Warning", "Railway crossing sign."),
    "t_intersection_l": ("t_junction_left", "Junction on left", "Warning", "Junction or intersection on the left."),
    "ped_zebra_cross": ("zebra_crossing", "Zebra crossing", "Information", "Marked pedestrian crossing."),
}

ALIASES = {"stop": "Stop", "do_not_enter": "NoEntry", "ped_crossing": "Pedestrians",
           "warning": "GeneralCaution"}
LIGHT_LABELS = {"traffic_light", "red_light", "green_light", "yellow_light"}

# Official GTSRB numeric order. The ResNet output order is stored in the manifest;
# it is alphabetical string order (0, 1, 10, ...), not numeric order.
GTSRB_LABELS = [
    "SpeedLimit20", "SpeedLimit30", "SpeedLimit50", "SpeedLimit60", "SpeedLimit70",
    "SpeedLimit80", "EndSpeedLimit80", "SpeedLimit100", "SpeedLimit120", "NoPassing",
    "NoPassingTrucks", "RightOfWayCrossing", "PriorityRoad", "Yield", "Stop", "NoVehicles",
    "VehiclesOver3.5TonsProhibited", "NoEntry", "GeneralCaution", "DangerousCurveLeft",
    "DangerousCurveRight", "DoubleCurve", "BumpyRoad", "SlipperyRoad", "RoadNarrowRight",
    "RoadWork", "TrafficSignals", "Pedestrians", "ChildrenCrossing", "BicyclesCrossing",
    "IceSnow", "WildAnimals", "EndAllRestrictions", "TurnRightAhead", "TurnLeftAhead",
    "AheadOnly", "GoStraightOrRight", "GoStraightOrLeft", "KeepRight", "KeepLeft",
    "RoundaboutMandatory", "EndNoPassing", "EndNoPassingByTrucks",
]


def sign_info(raw):
    key = ALIASES.get(raw, raw)
    if key not in SIGN_DEFINITIONS:
        raise ValueError(f"Unmapped model class: {raw}")
    identity, label, category, meaning = SIGN_DEFINITIONS[key]
    return {"id": identity, "label": label, "category": category, "meaning": meaning}


def catalogue():
    entries = [{"id": v[0], "label": v[1], "category": v[2], "meaning": v[3]}
               for v in SIGN_DEFINITIONS.values()]
    return sorted(entries, key=lambda row: (row["category"], row["label"]))
