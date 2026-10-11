"""One Curb, One Case — barrier taxonomy + PROWAG-grounded evidence rubric.

This module is the single source of truth for:

  1. TAXONOMY        — a Project-Sidewalk-style set of accessibility barrier
                       categories (CurbRamp, NoCurbRamp, Obstacle, SurfaceProblem,
                       NoSidewalk, Crosswalk, Signal).
  2. ISSUE_TYPE_MAP  — how the five Cambridge Open311 `issue_type` values map onto
                       those categories (text can refine the mapping at label time).
  3. RUBRIC          — a PER-CATEGORY required-evidence rubric. Each category names
                       the evidence fields a *complete* civic report needs so a
                       public-works crew can act, and — critically — the ONE
                       concrete follow-up instruction to emit when a field is
                       missing. This is the product thesis: tell the reporter the
                       single most useful measurement they still have to go take.
  4. GOLD_SCHEMA     — the compact gold object every labeled row conforms to.

Severity uses the Project Sidewalk 1-5 accessibility-impact scale
(1 = passable nuisance ... 5 = fully blocks a wheelchair/mobility-device user).
Project Sidewalk label types & severity: https://projectsidewalk.org/

PROWAG grounding
----------------
Dimensional thresholds in the rubric are grounded in the U.S. Access Board's
Public Right-of-Way Accessibility Guidelines (PROWAG), Technical Requirements:
    https://www.access-board.gov/prowag/technical.html
Key figures cited inline below (see each category):
  * R302.3  Pedestrian Access Route (PAR) continuous clear width >= 1.2 m (48 in);
            48 in is also the width that lets two canes / a wheelchair pass.
  * R304    Curb ramps: running slope <= 8.3% (1:12), cross slope <= 2% (1:48),
            with a detectable warning surface at the back of curb.
  * R208 /
    R305    Detectable warning surfaces (truncated domes) at curb ramps and at
            blended transitions / rail crossings.
  * R302.5  Surface — firm, stable, slip-resistant; vertical discontinuities
            > 1/4 in (6 mm) must be beveled; changes > 1/2 in are a barrier.
  * R302.7  Grade / running slope of the PAR.
  * R209 /
    R306    Accessible Pedestrian Signals (APS) — audible + vibrotactile
            indication, pushbutton locator tone, timing of the walk interval.
These are the numbers a reporter is told to go measure when they are missing.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# 1. TAXONOMY  (Project Sidewalk label types; https://projectsidewalk.org/)    #
# --------------------------------------------------------------------------- #

TAXONOMY: dict[str, str] = {
    "CurbRamp": "A curb ramp exists but is deficient (too steep, no detectable "
                "warning, broken lip, ponding). PROWAG R304.",
    "NoCurbRamp": "A curb ramp is absent where a pedestrian route meets the "
                  "street — forces a mobility-device user into traffic or a dead "
                  "end. PROWAG R304 (ramps required at crossings).",
    "Obstacle": "Something blocks the pedestrian access route — parked car, "
                "A-frame sign, trash/bulk item, overgrown vegetation, protruding "
                "object. Reduces or removes clear width. PROWAG R302.3 / R402.",
    "SurfaceProblem": "The walking surface is damaged or unsafe — heaved/cracked "
                      "brick or concrete, holes, tree-root uplift, ponding ice, "
                      "missing pavers. PROWAG R302.5 (firm/stable/level surface).",
    "NoSidewalk": "No sidewalk / pedestrian route exists along a segment where "
                  "one is needed. PROWAG R301 (PAR must be provided).",
    "Crosswalk": "A marked crossing is faded, missing, or unsafe to use — the "
                 "painted crossing itself, not its signal. PROWAG R302 (crossing "
                 "surface + markings).",
    "Signal": "A pedestrian/traffic signal or Accessible Pedestrian Signal (APS) "
              "is broken, mistimed, confusing, or physically damaged. PROWAG "
              "R209 / R306 (APS: audible + vibrotactile, walk-interval timing).",
}

# Project Sidewalk 1-5 severity scale (accessibility impact), cited verbatim in
# spirit from projectsidewalk.org severity guidance.
SEVERITY_SCALE: dict[int, str] = {
    1: "Passable — a minor nuisance; most users unaffected.",
    2: "Passable with care — slows or inconveniences some users.",
    3: "Difficult — a real obstacle for wheelchair / stroller / cane users; "
       "passable only with effort or help.",
    4: "Barely passable / unsafe — forces a dangerous workaround "
       "(e.g. into the street).",
    5: "Impassable — fully blocks a mobility-device user; no safe route.",
}


# --------------------------------------------------------------------------- #
# 2. ISSUE_TYPE_MAP  (Cambridge Open311 `issue_type` -> taxonomy default)       #
# --------------------------------------------------------------------------- #
# The city's five pedestrian issue_type values give a *default* category. The
# note text can refine it — e.g. a "Sidewalk Defect" that says "needs a curb
# ramp" is really NoCurbRamp, and one that says "no sloped entrance / lip" is a
# CurbRamp defect. Labelers (and the teacher model) use TEXT_REFINERS to override.

ISSUE_TYPE_MAP: dict[str, str] = {
    "Sidewalk Defect": "SurfaceProblem",
    "Sidewalk Obstruction": "Obstacle",
    "A-Frame/Business Sidewalk Obstruction": "Obstacle",
    "Botanical Obstruction": "Obstacle",
    "Traffic/Pedestrian Signal Complaint": "Signal",
}

# Lowercased substring -> category. Checked in order; first hit wins. Used only
# to *suggest* a refinement from the default issue_type mapping; a human/teacher
# makes the final call.
TEXT_REFINERS: list[tuple[str, str]] = [
    ("needs a curb ramp", "NoCurbRamp"),
    ("no curb ramp", "NoCurbRamp"),
    ("needs a ramp", "NoCurbRamp"),
    ("no sloped entrance", "NoCurbRamp"),
    ("not a sloped entrance", "NoCurbRamp"),
    ("no sidewalk", "NoSidewalk"),
    ("curb cut", "CurbRamp"),
    ("curb ramp", "CurbRamp"),
    ("handicap ramp", "CurbRamp"),
    ("h/c ramp", "CurbRamp"),
    ("ada ramp", "CurbRamp"),
    ("ada curb", "CurbRamp"),
    ("ramp lip", "CurbRamp"),
    ("crosswalk paint", "Crosswalk"),
    ("crosswalk is faded", "Crosswalk"),
    ("faded crosswalk", "Crosswalk"),
    ("detectable warning", "CurbRamp"),
]


def suggest_category(issue_type: str, description: str) -> str:
    """Return a *suggested* taxonomy category for a raw row.

    Starts from the issue_type default, then applies text refiners. This only
    suggests — the teacher model and the human labeler may override.
    """
    cat = ISSUE_TYPE_MAP.get(issue_type, "Obstacle")
    text = (description or "").lower()
    for needle, refined in TEXT_REFINERS:
        if needle in text:
            cat = refined
            break
    return cat


# --------------------------------------------------------------------------- #
# 3. RUBRIC  (per-category required evidence + the one follow-up per field)     #
# --------------------------------------------------------------------------- #
# Every category requires the same FOUR kinds of evidence a crew needs to act:
#   location    — a findable spot (cross-street / frontage / which side / which
#                 corner). Without it the crew cannot find the barrier.
#   measurement — the category-specific dimension that decides compliance,
#                 grounded in a PROWAG threshold.
#   impact      — the affected user and how (wheelchair / stroller / cane), i.e.
#                 why this is an accessibility barrier, not just a complaint.
#   persistence — how long it has been there / whether it is permanent, so the
#                 city can triage a one-off vs. a standing barrier.
#
# For each field we store:
#   label       — human name of the evidence field
#   why         — what decision it drives / PROWAG hook
#   followup    — the ONE concrete instruction to emit if the field is missing
#                 (imperative, specific, measurable — "go do this").
# `followup_priority` orders the fields: when several are missing, the gold
# `followup_instruction` is the follow-up of the highest-priority missing field.

EvidenceField = dict[str, str]


def _loc(followup: str) -> EvidenceField:
    return {
        "field": "location",
        "label": "precise location",
        "why": "A crew must be able to find the exact spot (which corner / side / "
               "frontage / cross-street), not just the block.",
        "followup": followup,
    }


def _impact(followup: str) -> EvidenceField:
    return {
        "field": "impact",
        "label": "impact on a wheelchair / stroller / cane user",
        "why": "Confirms this is an accessibility barrier and sets severity "
               "(Project Sidewalk 1-5).",
        "followup": followup,
    }


def _persist(followup: str) -> EvidenceField:
    return {
        "field": "persistence",
        "label": "persistence (how long present / permanent?)",
        "why": "Lets the city triage a transient obstruction vs. a standing "
               "built-environment barrier.",
        "followup": followup,
    }


# Per-category required-evidence lists. `followup_priority` lists field names
# worst-first (missing measurement usually matters most for action).
RUBRIC: dict[str, dict] = {
    "CurbRamp": {
        "prowag": "R304 — running slope <= 8.3% (1:12), cross slope <= 2%, "
                  "detectable warning at back of curb; R302.5 — vertical lip "
                  "> 1/4 in must be beveled.",
        "required_evidence": [
            _loc("Name the exact corner and which ramp (e.g. 'SE corner of "
                 "Windsor & State, ramp facing State St')."),
            {
                "field": "measurement",
                "label": "ramp running slope AND/OR vertical lip height at the "
                         "bottom edge",
                "why": "PROWAG R304 caps running slope at 8.3% and R302.5 "
                       "requires lips > 1/4 in to be beveled; these decide "
                       "whether the ramp is compliant.",
                "followup": "Measure the vertical lip at the bottom edge of the "
                            "ramp with a ruler, in inches (anything over 1/4 in "
                            "is a barrier); if you can, lay a phone level on the "
                            "slope and note the running grade.",
            },
            {
                "field": "detectable_warning",
                "label": "presence/condition of the detectable warning "
                         "(truncated-dome) surface",
                "why": "PROWAG R304/R305 require a detectable warning so a cane "
                       "user can feel the street edge.",
                "followup": "Check the back of the ramp for the bumpy "
                            "truncated-dome pad; note whether it is present, "
                            "missing, or worn flat.",
            },
            _impact("Note who is blocked and how — e.g. 'a wheelchair cannot "
                    "mount the lip and has to roll into the crossing'."),
            _persist("Note whether this is a permanent build issue or recent "
                     "damage, and how long it has been like this."),
        ],
        "followup_priority": ["measurement", "location",
                              "detectable_warning", "impact", "persistence"],
    },
    "NoCurbRamp": {
        "prowag": "R304 — a curb ramp (or blended transition) is required where "
                  "a pedestrian access route crosses a curb.",
        "required_evidence": [
            _loc("Name the exact corner/crossing missing the ramp and the "
                 "direction of travel (e.g. 'NW corner of Norfolk & School, "
                 "crossing toward the Recycling Center')."),
            {
                "field": "measurement",
                "label": "curb height a user must drop/climb, and whether ANY "
                         "ramp exists on that corner",
                "why": "PROWAG R304 requires a ramp at the crossing; the curb "
                       "height sets how impassable the gap is.",
                "followup": "Measure the curb face height in inches at the point "
                            "where the sidewalk meets the street, and confirm "
                            "there is no ramp on that corner at all.",
            },
            _impact("Note how a wheelchair/stroller is forced to detour or enter "
                    "traffic because there is no sloped entrance."),
            _persist("Confirm this is a permanent missing-ramp condition, not a "
                     "temporarily blocked ramp."),
        ],
        "followup_priority": ["location", "measurement", "impact", "persistence"],
    },
    "Obstacle": {
        "prowag": "R302.3 — pedestrian access route continuous clear width must "
                  "be >= 1.2 m (48 in); R402 — protruding objects.",
        "required_evidence": [
            _loc("Give the frontage/address and which part of the walk is "
                 "blocked (e.g. 'full width in front of 115 Chilton St')."),
            {
                "field": "measurement",
                "label": "remaining clear width past the obstruction (inches)",
                "why": "PROWAG R302.3 requires >= 48 in clear; the remaining "
                       "width decides whether a wheelchair can still pass.",
                "followup": "Measure the clear width still open past the "
                            "obstruction in inches (PROWAG needs at least 48 in / "
                            "1.2 m); if under 36 in say so explicitly.",
            },
            _impact("Note who cannot get by and what they must do instead "
                    "(e.g. 'strollers forced into traffic')."),
            _persist("State how long the obstruction has been there and whether "
                     "it recurs (one-off dumping vs. a standing A-frame)."),
        ],
        "followup_priority": ["measurement", "location", "impact", "persistence"],
    },
    "SurfaceProblem": {
        "prowag": "R302.5 — surface firm, stable, slip-resistant; vertical "
                  "discontinuities > 1/4 in (6 mm) beveled; R302.7 — grade.",
        "required_evidence": [
            _loc("Pin the defect to a frontage/landmark and which stretch "
                 "(e.g. 'between the tree wells at 315 Harvard St')."),
            {
                "field": "measurement",
                "label": "size of the defect — vertical lip/heave height, hole "
                         "depth, or length of broken stretch (inches)",
                "why": "PROWAG R302.5 treats vertical changes > 1/4 in as a "
                       "barrier (> 1/2 in impassable); size sets severity.",
                "followup": "Measure the defect: the height of the vertical lip "
                            "or the depth of the hole in inches, and the length "
                            "of the broken stretch in feet.",
            },
            {
                "field": "clear_width",
                "label": "clear width still usable alongside the defect",
                "why": "A defect that leaves < 48 in (PROWAG R302.3) usable is "
                       "far worse for a wheelchair.",
                "followup": "Measure how much level, usable sidewalk width "
                            "remains beside the defect, in inches.",
            },
            _impact("Note the mobility impact — e.g. 'stroller wheels catch', "
                    "'trip hazard for a cane user', 'ice sheet in winter'."),
            _persist("Note how long the defect has existed / whether it is "
                     "worsening (eroding hole, spreading cracks)."),
        ],
        "followup_priority": ["measurement", "location", "clear_width",
                              "impact", "persistence"],
    },
    "NoSidewalk": {
        "prowag": "R301 — a pedestrian access route must be provided along a "
                  "pedestrian circulation path.",
        "required_evidence": [
            _loc("Name the segment with no sidewalk by its two endpoints "
                 "(from/to cross-streets) and which side of the street."),
            {
                "field": "measurement",
                "label": "length of the gap and what pedestrians walk on instead",
                "why": "PROWAG R301 requires a route; the gap length and the "
                       "surface (dirt shoulder, roadway) set the hazard.",
                "followup": "Pace or measure the length of the no-sidewalk gap in "
                            "feet and note what pedestrians currently walk on "
                            "(road shoulder, dirt, grass).",
            },
            _impact("Note how a wheelchair/stroller user is forced into the road "
                    "and the traffic speed/volume there."),
            _persist("Confirm this is a permanent gap in the network, not a "
                     "temporary closure."),
        ],
        "followup_priority": ["location", "measurement", "impact", "persistence"],
    },
    "Crosswalk": {
        "prowag": "R302 — crossing surface and markings; detectable warnings at "
                  "the crossing where applicable.",
        "required_evidence": [
            _loc("Name the two streets of the crossing and which leg "
                 "(e.g. 'the Acorn St leg of the Acorn & Mass Ave crossing')."),
            {
                "field": "measurement",
                "label": "condition/visibility of the markings and crossing "
                         "length",
                "why": "Faded or missing markings make the crossing unsafe; "
                       "length matters for signal timing (PROWAG R306).",
                "followup": "Note how faded the paint is (e.g. 'bars 50% gone') "
                            "and roughly how many feet the crossing spans.",
            },
            _impact("Note the risk to pedestrians, especially low-vision users "
                    "who rely on clear markings."),
            _persist("Note how long the markings have been faded/absent."),
        ],
        "followup_priority": ["measurement", "location", "impact", "persistence"],
    },
    "Signal": {
        "prowag": "R209 / R306 — Accessible Pedestrian Signals: audible + "
                  "vibrotactile walk indication, pushbutton locator tone, and "
                  "walk-interval timing adequate for the crossing length.",
        "required_evidence": [
            _loc("Name the intersection and which crossing/pushbutton "
                 "(e.g. 'the Mass Ave push-button on the Masonic-temple side')."),
            {
                "field": "measurement",
                "label": "what exactly is wrong — dead pushbutton, no "
                         "audible/vibrotactile cue, wrong timing, physical "
                         "damage — and since when",
                "why": "PROWAG R209/R306 require working APS cues and adequate "
                       "walk timing; the specific failure drives the fix.",
                "followup": "At the signal, press the pedestrian button and note "
                            "exactly what fails: no walk light, no audible/"
                            "vibrotactile cue, button loose/dead, or too-short "
                            "walk time (count the seconds).",
            },
            _impact("Note who is endangered — e.g. a low-vision user with no "
                    "audible cue, or anyone given too little crossing time."),
            _persist("Note how long it has malfunctioned (e.g. 'at least a "
                     "week')."),
        ],
        "followup_priority": ["measurement", "location", "impact", "persistence"],
    },
}


# --------------------------------------------------------------------------- #
# 4. GOLD_SCHEMA  (the compact object every labeled row conforms to)            #
# --------------------------------------------------------------------------- #
# Single source of truth for both the teacher-labeled TRAIN gold and the
# human-labeled HELD-OUT gold.
#
#   category             : one TAXONOMY key
#   severity             : int 1-5 (Project Sidewalk scale)
#   objective_description: neutral, factual restatement of the barrier (no
#                          editorializing), suitable for a civic work order
#   location             : the findable location string a crew would use
#   missing_evidence     : list[str] — which required-evidence fields the note
#                          does NOT already supply (by `field` name from RUBRIC)
#   followup_instruction : the single highest-priority follow-up, derived from
#                          the highest `followup_priority` field present in
#                          missing_evidence. Empty when missing_evidence is empty.

GOLD_FIELDS: list[str] = [
    "category",
    "severity",
    "objective_description",
    "location",
    "missing_evidence",
    "followup_instruction",
]

GOLD_SCHEMA: dict = {
    "category": "str (one of TAXONOMY keys)",
    "severity": "int 1-5 (SEVERITY_SCALE)",
    "objective_description": "str — neutral factual restatement",
    "location": "str — findable location for a crew",
    "missing_evidence": "list[str] — RUBRIC field names not supplied by the note",
    "followup_instruction": "str — single highest-priority follow-up (or '')",
}


def followup_for(category: str, missing_evidence: list[str]) -> str:
    """Derive the single `followup_instruction` from a category + missing list.

    Picks the highest-priority missing field (per the category's
    `followup_priority`) and returns its `followup` text. Empty missing list
    yields an empty string. This keeps `followup_instruction` deterministic and
    always consistent with `missing_evidence`.
    """
    if category not in RUBRIC:
        return ""
    if not missing_evidence:
        return ""
    spec = RUBRIC[category]
    priority = spec["followup_priority"]
    missing_set = set(missing_evidence)
    # fields in priority order that are actually missing
    for field in priority:
        if field in missing_set:
            for ev in spec["required_evidence"]:
                if ev["field"] == field:
                    return ev["followup"]
    # a missing field not in priority list (shouldn't happen) — fall back
    first = missing_evidence[0]
    for ev in spec["required_evidence"]:
        if ev["field"] == first:
            return ev["followup"]
    return ""


def required_fields(category: str) -> list[str]:
    """Return the list of required-evidence field names for a category."""
    spec = RUBRIC.get(category)
    if not spec:
        return []
    return [ev["field"] for ev in spec["required_evidence"]]


def validate_gold_object(obj: dict) -> list[str]:
    """Validate one gold object against the schema + taxonomy. Return error list
    (empty = valid)."""
    errs: list[str] = []
    for f in GOLD_FIELDS:
        if f not in obj:
            errs.append(f"missing field: {f}")
    if errs:
        return errs
    if obj["category"] not in TAXONOMY:
        errs.append(f"category not in taxonomy: {obj['category']!r}")
    sev = obj["severity"]
    if not isinstance(sev, int) or sev < 1 or sev > 5:
        errs.append(f"severity must be int 1-5, got {sev!r}")
    if not isinstance(obj["objective_description"], str) or not obj["objective_description"].strip():
        errs.append("objective_description must be a non-empty string")
    if not isinstance(obj["location"], str):
        errs.append("location must be a string")
    me = obj["missing_evidence"]
    if not isinstance(me, list) or not all(isinstance(x, str) for x in me):
        errs.append("missing_evidence must be a list[str]")
    else:
        allowed = set(required_fields(obj["category"])) if obj["category"] in RUBRIC else set()
        for field in me:
            if field not in allowed:
                errs.append(f"missing_evidence item {field!r} not a required field "
                            f"for {obj['category']}")
    if not isinstance(obj["followup_instruction"], str):
        errs.append("followup_instruction must be a string")
    else:
        if not me and obj["followup_instruction"]:
            errs.append("followup_instruction must be '' when missing_evidence is empty")
        if me and not obj["followup_instruction"]:
            errs.append("followup_instruction must be non-empty when evidence is missing")
    return errs


def validate_self() -> None:
    """Self-check the rubric is internally consistent. Raises on any problem."""
    problems: list[str] = []

    # every taxonomy category has a rubric entry
    for cat in TAXONOMY:
        if cat not in RUBRIC:
            problems.append(f"taxonomy category {cat!r} has no RUBRIC entry")

    # every rubric entry maps to a real taxonomy category + is well-formed
    for cat, spec in RUBRIC.items():
        if cat not in TAXONOMY:
            problems.append(f"RUBRIC category {cat!r} not in TAXONOMY")
        if "prowag" not in spec or not spec["prowag"]:
            problems.append(f"{cat}: missing PROWAG citation")
        fields = [ev["field"] for ev in spec["required_evidence"]]
        # the four core evidence kinds must be present
        for core in ("location", "measurement", "impact", "persistence"):
            if core not in fields:
                problems.append(f"{cat}: required core evidence {core!r} absent")
        # every required field has a non-empty follow-up template
        for ev in spec["required_evidence"]:
            if not ev.get("followup"):
                problems.append(f"{cat}:{ev['field']} has no follow-up template")
        # followup_priority references only real fields, and vice versa
        for pf in spec["followup_priority"]:
            if pf not in fields:
                problems.append(f"{cat}: followup_priority has unknown field {pf!r}")
        for f in fields:
            if f not in spec["followup_priority"]:
                problems.append(f"{cat}: field {f!r} absent from followup_priority")

    # issue_type map targets must be real categories
    for it, cat in ISSUE_TYPE_MAP.items():
        if cat not in TAXONOMY:
            problems.append(f"ISSUE_TYPE_MAP[{it!r}] -> unknown category {cat!r}")
    for _needle, cat in TEXT_REFINERS:
        if cat not in TAXONOMY:
            problems.append(f"TEXT_REFINERS -> unknown category {cat!r}")

    # followup_for is consistent: non-empty for any single missing field
    for cat, spec in RUBRIC.items():
        for ev in spec["required_evidence"]:
            fi = followup_for(cat, [ev["field"]])
            if not fi:
                problems.append(f"followup_for({cat}, [{ev['field']}]) empty")
        if followup_for(cat, []) != "":
            problems.append(f"followup_for({cat}, []) should be ''")

    if problems:
        raise AssertionError("rubric self-check failed:\n  - " + "\n  - ".join(problems))


if __name__ == "__main__":
    validate_self()
    print("rubric.py self-check OK")
    print(f"  taxonomy categories : {len(TAXONOMY)}")
    print(f"  rubric categories   : {len(RUBRIC)}")
    print(f"  issue_type map keys : {len(ISSUE_TYPE_MAP)}")
    print(f"  gold schema fields  : {GOLD_FIELDS}")
