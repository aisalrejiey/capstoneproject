"""Who raised a feedback entry: role + number, e.g. client2, user1, developer3, tester4.
The role changes how much weight the entry gets in the priority score."""
import os
import re
from datetime import datetime

ROLES = ["user", "client", "developer", "tester"]
DEFAULT_WEIGHTS = {"client": 1.5, "user": 1.0, "tester": 0.8, "developer": 0.7}
_PAT = re.compile(r"^\s*(user|client|developer|tester)\s*[-_ ]?\s*(\d{1,3})?\s*$", re.I)


def weights():
    """ROLE_WEIGHTS in .env, e.g. client=1.5,user=1.0,tester=0.8,developer=0.7"""
    w = dict(DEFAULT_WEIGHTS)
    for part in os.getenv("ROLE_WEIGHTS", "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                if k.strip().lower() in ROLES:
                    w[k.strip().lower()] = max(0.0, float(v))
            except ValueError:
                pass
    return w


def make(role, number):
    return f"{role}{int(number)}"


def parse_user(s):
    """'Client 2' / 'client-2' / 'client2' -> 'client2'.  blank -> ''.  unrecognised -> None."""
    if s is None or str(s).strip() == "" or str(s).strip().lower() == "nan":
        return ""
    m = _PAT.match(str(s))
    if not m:
        return None
    return f"{m.group(1).lower()}{int(m.group(2))}" if m.group(2) else m.group(1).lower()


def role_of(user):
    m = re.match(r"[a-z]+", user or "")
    return m.group(0) if m and m.group(0) in ROLES else ""


def weight(user):
    """Priority weight for whoever raised it (1.0 if unspecified)."""
    return weights().get(role_of(user), 1.0)


def adj_urgency(urgency, user):
    """Model urgency (1-5) scaled by the raiser's weight, capped at 5."""
    return round(min(5.0, float(urgency) * weight(user)), 1)


# ---------- urgency boosts: age and repeats ----------
def _int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def age_boost(created_at):
    """+5% per full 10 days since the entry's date, max +30% (all configurable). Returns 0.0 - 0.30."""
    if not created_at:
        return 0.0
    days = max((datetime.now() - datetime.fromisoformat(created_at)).total_seconds() / 86400, 0.0)
    step = max(_int("AGE_STEP_DAYS", 10), 1)
    return min(_int("AGE_BOOST_PCT_PER_STEP", 5) / 100 * int(days // step),
               _int("AGE_BOOST_MAX_PCT", 30) / 100)


def repeat_boost(n):
    """+5% for each extra request on the same theme (from anyone), max +20%. Returns 0.0 - 0.20."""
    return min(_int("REPEAT_BOOST_PCT_EACH", 5) / 100 * max(int(n) - 1, 0),
               _int("REPEAT_BOOST_MAX_PCT", 20) / 100)


def final_urgency(urgency, user, created_at, n_in_theme):
    """Model urgency (1-5) x role weight x (1 + age boost + repeat boost), capped at 5 (= 100%)."""
    base = float(urgency) * weight(user)
    return min(5.0, base * (1 + age_boost(created_at) + repeat_boost(n_in_theme)))


def pct(u):
    """Urgency on the 0-5 scale as a percentage, never above 100."""
    return round(min(100.0, u / 5 * 100), 1)
