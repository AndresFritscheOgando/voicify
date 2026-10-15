"""Number and abbreviation fixes for what Kokoro's G2P (misaki) reads wrong.

Each rule exists because misaki got the case wrong when probed in Phase 8 (see docs/phases.md).
Years, ordinals, currency, percentages, decimals, Dr./Mr./Mrs./Ms., "St." before a name, and
common acronyms are already read correctly and are left alone. Turn this off with
TTS_NORMALIZE=false.
"""

from __future__ import annotations

import re

from tts_agent.processing.spoken import Rule

# --- abbreviations ---------------------------------------------------------------------------

# Read as letters or clipped words ("Prof." -> "prof", "Lt." -> "L T"), and the period becomes a
# sentence pause. Only expanded before a capitalized name.
_TITLES = {
    "Prof": "Professor",
    "Gov": "Governor",
    "Gen": "General",
    "Capt": "Captain",
    "Lt": "Lieutenant",
    "Sgt": "Sergeant",
    "Rev": "Reverend",
    "Hon": "Honorable",
    "Mt": "Mount",
    "Ft": "Fort",
}
# Expanded after a capitalized word or a number ("Main St.", "5th Ave.").
_STREETS = {"St": "Street", "Ave": "Avenue", "Rd": "Road", "Blvd": "Boulevard"}
# Expanded after a capitalized name ("John Smith Jr.").
_SUFFIXES = {"Jr": "Junior", "Sr": "Senior"}
# Expanded anywhere.
_REFERENCES = {
    "Fig": "Figure",
    "Figs": "Figures",
    "Vol": "Volume",
    "Ch": "Chapter",
    "Sec": "Section",
    "Eq": "Equation",
    "pp": "pages",
    "Dept": "Department",
    "Univ": "University",
    "Approx": "Approximately",
    "approx": "approximately",
    "Inc": "Incorporated",
    "Corp": "Corporation",
}

_PREVIOUS_WORD = re.compile(r"(\S+)\s+$")
_NEXT_WORD = re.compile(r"\s+(\S+)")


def _previous_word(m: re.Match[str]) -> str:
    found = _PREVIOUS_WORD.search(m.string, 0, m.start())
    return found.group(1) if found else ""


def _next_word(m: re.Match[str]) -> str:
    found = _NEXT_WORD.match(m.string, m.end())
    return found.group(1) if found else ""


def _capitalized(word: str) -> bool:
    return word[:1].isupper()


def _ends_sentence(m: re.Match[str]) -> bool:
    """True if the abbreviation's period probably also ends the sentence."""
    rest = m.string[m.end() :].lstrip()
    return not rest or rest[0].isupper() or rest[0] in "\"'("


def _expanded(word: str, m: re.Match[str]) -> str:
    return f"{word}." if _ends_sentence(m) else word


def _abbreviation(m: re.Match[str]) -> str | None:
    abbr = m.group(1)
    before, after = _previous_word(m), _next_word(m)
    if abbr == "St":
        if _capitalized(after) and not (_capitalized(before) or before[:1].isdigit()):
            return "Saint"
        if _capitalized(before) or before[:1].isdigit():
            return _expanded("Street", m)
        return None
    if abbr in _TITLES:
        return _TITLES[abbr] if _capitalized(after) else None
    if abbr in _STREETS:
        named = _capitalized(before) or before[:1].isdigit()
        return _expanded(_STREETS[abbr], m) if named else None
    if abbr in _SUFFIXES:
        return _expanded(_SUFFIXES[abbr], m) if _capitalized(before.rstrip(",")) else None
    if abbr in _REFERENCES:
        return _expanded(_REFERENCES[abbr], m)
    return None


_ABBREVIATION_WORDS = sorted({*_TITLES, *_STREETS, *_SUFFIXES, *_REFERENCES}, key=len, reverse=True)
ABBREVIATIONS = Rule(
    "abbreviation",
    re.compile(rf"(?<![\w.])({'|'.join(_ABBREVIATION_WORDS)})\."),
    _abbreviation,
)

# --- clock times -----------------------------------------------------------------------------
# misaki keeps the colon and reads "10:00" as "ten zero zero"; "7am" as "seven am".


def _meridiem(m: re.Match[str], letter: str | None, dotted: str, final_dot: str) -> str:
    """ " AM"/" PM", keeping a final period that also ends the sentence ("at 3 p.m." / "3 pm.")."""
    if not letter:
        return ""
    period = "." if final_dot and (not dotted or _ends_sentence(m)) else ""
    return f" {letter.upper()}M{period}"


def _time(m: re.Match[str]) -> str | None:
    hour, minute = int(m.group(1)), m.group(2)
    if hour > 24 or int(minute) > 59:
        return None
    suffix = _meridiem(m, m.group(3), m.group(4), m.group(5))
    if minute == "00":
        return f"{hour}{suffix}" if suffix else f"{hour} o'clock"
    if minute.startswith("0"):
        return f"{hour} oh {minute[1]}{suffix}"
    return f"{hour} {minute}{suffix}"


_MERIDIEM = r"(?:\s?([AaPp])(\.?)[Mm](\.?)(?![A-Za-z]))"
TIMES = Rule(
    "time",
    re.compile(rf"(?<![\d:])(\d{{1,2}}):(\d{{2}}){_MERIDIEM}?(?![\d:])"),
    _time,
)
COMPACT_TIMES = Rule(
    "compact time",
    re.compile(r"(?<![\w.])(\d{1,2})([AaPp])[Mm](?![A-Za-z])"),
    lambda m: f"{m.group(1)} {m.group(2).upper()}M" if int(m.group(1)) <= 12 else None,
)

# --- phone numbers ---------------------------------------------------------------------------
# misaki reads "555-1234" as "five hundred fifty five, twelve thirty four".


def _digits(m: re.Match[str]) -> str:
    groups = re.findall(r"\d+", m.group())
    return ", ".join(" ".join(group) for group in groups)


PHONE_NUMBERS = Rule(
    "phone number",
    re.compile(
        r"(?<![\w-])(?:1-)?(?:\(\d{3}\)\s?|\d{3}-)\d{3}-\d{4}(?![\w-])"
        r"|(?<![\w-])[2-9]\d{2}-\d{4}(?![\w-])"
    ),
    _digits,
)

# --- ranges ----------------------------------------------------------------------------------
# misaki reads "1-10" as "one-ten" and "10–20" as "ten twenty".

RANGES = Rule(
    "range",
    re.compile(
        r"(?<![\w:/\-–])(?<!\d[.,])(\d+(?:\.\d+)?)\s?[-–]\s?(\d+(?:\.\d+)?)"
        r"(?![\w:/\-–])(?![.,]\d)"
    ),
    lambda m: f"{m.group(1)} to {m.group(2)}",
)

# --- units after a number --------------------------------------------------------------------
# misaki spells most of these out as letters ("5 km" -> "five K M").

_UNITS = {
    "km/h": ("kilometer per hour", "kilometers per hour"),
    "km": ("kilometer", "kilometers"),
    "cm": ("centimeter", "centimeters"),
    "mm": ("millimeter", "millimeters"),
    "kg": ("kilogram", "kilograms"),
    "mg": ("milligram", "milligrams"),
    "g": ("gram", "grams"),
    "lb": ("pound", "pounds"),
    "oz": ("ounce", "ounces"),
    "ft": ("foot", "feet"),
    "mi": ("mile", "miles"),
    "mph": ("mile per hour", "miles per hour"),
    "ml": ("milliliter", "milliliters"),
    "mL": ("milliliter", "milliliters"),
    "L": ("liter", "liters"),
    "KB": ("kilobyte", "kilobytes"),
    "MB": ("megabyte", "megabytes"),
    "GB": ("gigabyte", "gigabytes"),
    "TB": ("terabyte", "terabytes"),
    "°C": ("degree Celsius", "degrees Celsius"),
    "°F": ("degree Fahrenheit", "degrees Fahrenheit"),
}
_UNIT_NAMES = sorted(_UNITS, key=len, reverse=True)


def _unit(m: re.Match[str]) -> str:
    number, unit = m.group(1), m.group(2)
    singular, plural = _UNITS[unit]
    # "a 5 GB file" is read "a five gigabyte file"
    as_adjective = _previous_word(m).lower() in {"a", "an"}
    return f"{number} {singular if number == '1' or as_adjective else plural}"


UNITS = Rule(
    "unit",
    re.compile(
        rf"(?<![\w.])(\d+(?:[.,]\d+)*)\s?({'|'.join(re.escape(u) for u in _UNIT_NAMES)})"
        r"(?![\w/°])"
    ),
    _unit,
)

# --- Roman numerals --------------------------------------------------------------------------
# misaki spells "World War II" as "I I" and "Henry VIII" as "V I I I".

# Words after which a numeral is a count ("World War 2"), not a regnal number ("Henry the 8th").
_COUNTED_WORDS = """
    War Wars Chapter Part Book Volume Act Scene Phase Episode Season Level Bowl Apollo Rocky
    Grade Class Type Stage Round Mark Title Article Section Appendix Figure Table Vatican Fantasy
"""
_COUNTED = frozenset(_COUNTED_WORDS.split())
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10}


def _roman_value(numeral: str) -> int | None:
    total = 0
    for i, char in enumerate(numeral):
        value = _ROMAN_VALUES[char]
        following = _ROMAN_VALUES[numeral[i + 1]] if i + 1 < len(numeral) else 0
        total += -value if value < following else value
    return total if 2 <= total <= 39 and _to_roman(total) == numeral else None


def _to_roman(value: int) -> str:
    out = ""
    for amount, symbol in ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while value >= amount:
            out += symbol
            value -= amount
    return out


def _ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _roman(m: re.Match[str]) -> str | None:
    word, numeral = m.group(1), m.group(2)
    value = _roman_value(numeral)
    if value is None:
        return None
    if word in _COUNTED:
        return f"{word} {value}"
    return f"{word} the {_ordinal(value)}"  # a monarch or pope: "Henry the 8th"


ROMAN_NUMERALS = Rule(
    "roman numeral",
    re.compile(r"(?<![\w'])([A-Z][a-z]+) ([IVX]{2,6})(?![\w'])"),
    _roman,
)

# --- fractions, multipliers, "#1" ------------------------------------------------------------
# misaki reads "1/2" as "one two", "5x" as "five ex", and drops the "#" in "#1".

_FRACTIONS = {
    "1/2": "one half",
    "1/3": "one third",
    "2/3": "two thirds",
    "1/4": "one quarter",
    "3/4": "three quarters",
}
FRACTIONS = Rule(
    "fraction",
    re.compile(r"(?<![\d/.])(?:(\d+) )?(1/2|1/3|2/3|1/4|3/4)(?![\d/])"),
    lambda m: (f"{m.group(1)} and " if m.group(1) else "") + _FRACTIONS[m.group(2)],
)
MULTIPLIERS = Rule(
    "multiplier",
    re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)x(?!\w)"),
    lambda m: f"{m.group(1)} times",
)
NUMBER_SIGNS = Rule(
    "number sign",
    re.compile(r"(?<![\w#])#(\d+)(?!\w)"),
    lambda m: f"number {m.group(1)}",
)

# Order matters where patterns could overlap: earlier rules claim their text first.
NORMALIZATION_RULES: tuple[Rule, ...] = (
    PHONE_NUMBERS,
    TIMES,
    COMPACT_TIMES,
    UNITS,
    RANGES,
    FRACTIONS,
    MULTIPLIERS,
    NUMBER_SIGNS,
    ABBREVIATIONS,
    ROMAN_NUMERALS,
)
