"""Propeller vs jet classification from ICAO type designators.

Like the reference method, we drop *known jets* and keep everything else
(props, helicopters and unknown types), so missing type data never hides the
right aircraft. Pass an ICAO Doc 8643 export for exact engine types.
"""

import pandas as pd

JET_TYPES = {
    # Airbus
    "A318", "A319", "A320", "A321", "A19N", "A20N", "A21N", "A332", "A333", "A338", "A339",
    "A342", "A343", "A345", "A346", "A359", "A35K", "A388", "A306", "A30B", "A310", "BCS1", "BCS3",
    # Boeing
    "B712", "B717", "B721", "B722", "B732", "B733", "B734", "B735", "B736", "B737", "B738", "B739",
    "B37M", "B38M", "B39M", "B3XM", "B752", "B753", "B762", "B763", "B764", "B772", "B773", "B77L",
    "B77W", "B778", "B779", "B788", "B789", "B78X", "B742", "B743", "B744", "B748", "B74S",
    "MD11", "MD81", "MD82", "MD83", "MD87", "MD88", "MD90", "DC10", "DC93", "DC95",
    # Regional jets
    "CRJ1", "CRJ2", "CRJ7", "CRJ9", "CRJX", "E135", "E145", "E35L", "E45X", "E170", "E75L", "E75S",
    "E190", "E195", "E290", "E295", "F70", "F100", "RJ70", "RJ85", "RJ1H", "B461", "B462", "B463",
    # Business jets
    "C500", "C501", "C510", "C525", "C25A", "C25B", "C25C", "C25M", "C526", "C550", "C551", "C55B",
    "C560", "C56X", "C650", "C680", "C68A", "C700", "C750", "CL30", "CL35", "CL60", "GL5T", "GL7T",
    "GLEX", "GLF2", "GLF3", "GLF4", "GLF5", "GLF6", "GA5C", "GA6C", "GA7C", "GA8C", "G150", "G280",
    "LJ23", "LJ24", "LJ25", "LJ31", "LJ35", "LJ40", "LJ45", "LJ55", "LJ60", "LJ70", "LJ75",
    "E50P", "E55P", "E545", "E550", "PC24", "HDJT", "SF50", "EA50", "F2TH", "FA10", "FA20", "FA50",
    "FA6X", "FA7X", "FA8X", "F900", "H25A", "H25B", "H25C", "BE40", "PRM1", "ASTR", "GALX", "WW24",
    "J328", "D328", "ECJ", "C30J",
    # Military / other common jets
    "C17", "C5M", "K35R", "KC46", "E3TF", "E6", "P8", "F16", "F15", "F18", "F18S", "F35", "F22",
    "T38", "T6", "A10", "B52", "B1", "B2", "C32", "C40", "C37A", "C37B", "GLF5",
}
# T6 (Texan II) is a turboprop; keep it in the prop class.
JET_TYPES.discard("T6")

PROP_ENGINE_TYPES = {"piston", "turboprop", "turboshaft", "turboprop/turboshaft", "electric", "p", "t", "e"}


def load_type_db(path):
    """Read a CSV with columns Designator + EngineType (ICAO Doc 8643 export)."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    des = cols.get("designator") or cols.get("typecode") or cols.get("type")
    eng = cols.get("enginetype") or cols.get("engine_type") or cols.get("engine")
    return {str(d).upper(): str(e).strip().lower() for d, e in zip(df[des], df[eng])}


def is_propeller(typecode, type_db=None):
    tc = (typecode or "").strip().upper()
    if not tc:
        return True
    if type_db and tc in type_db:
        return type_db[tc] in PROP_ENGINE_TYPES
    return tc not in JET_TYPES
