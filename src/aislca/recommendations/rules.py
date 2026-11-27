"""Rule-based weakness detection and recommendations (Module 4).

These are plain "if this, then suggest that" rules. Every recommendation
has a plain-language "why". The dataset has no topic-level marks, so the
advice is at subject and habit level. All limits come from
configs/config.yaml, section "recommendations".

Recommendations are guidance, not guarantees.
"""

from aislca.data.ingestion import load_config

# Smaller number = shown first
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}

# UCI study-time codes (WEEKLY study time)
STUDY_TIME_LABELS = {
    1: "less than 2 hours",
    2: "2 to 5 hours",
    3: "5 to 10 hours",
    4: "more than 10 hours",
}

DISCLAIMER = (
    "These are suggestions to support your own plan and your teachers' advice. "
    "They are guidance, not guarantees."
)


def load_rule_settings(config=None):
    """Read the limits from the config file."""
    config = config or load_config()
    settings = dict(config["recommendations"])
    settings["pass_mark"] = config["target"]["pass_mark"]
    return settings


def mark_band(mark, settings):
    """Turn a mark (0 to 20) into a performance band."""
    if mark < settings["weak_mark_below"]:
        return "Needs improvement"
    if mark < settings["moderate_mark_below"]:
        return "Moderate"
    if mark < settings["strong_mark_from"]:
        return "Good"
    return "Strong"


def subject_table(marks, settings):
    """A small table: one row per subject with its mark and band."""
    rows = []
    for subject, mark in marks.items():
        rows.append({
            "subject": subject,
            "mark": mark,
            "performance": mark_band(mark, settings),
        })
    return rows


def make_recommendation(rule, area, priority, action, why, goal=""):
    """Build one recommendation as a dictionary."""
    return {
        "rule": rule,
        "area": area,
        "priority": priority,
        "action": action,
        "why": why,
        "goal": goal,
    }


def priority_rank(recommendation):
    """Used to sort: high first, then medium, low, info."""
    return PRIORITY_ORDER[recommendation["priority"]]


def model_rule(fail_probability, support_threshold):
    """Suggest extra support when the model flags the student (Math model)."""
    if fail_probability is None or support_threshold is None:
        return []
    if fail_probability < support_threshold:
        return []

    percent = fail_probability * 100
    return [make_recommendation(
        rule="model_flag",
        area="Mathematics",
        priority="high",
        action="Talk to a teacher or school counselor about what extra support would help.",
        why=(f"The prediction model's estimated probability of not passing Mathematics is "
             f"{percent:.0f}%, which is at or above the flag threshold of "
             f"{support_threshold:g}. This is an estimate based on past students, "
             "not a certainty."),
    )]


def subject_rules(marks, settings):
    """One rule per subject, based on the current mark (0 to 20)."""
    recommendations = []
    sessions = settings["practice_sessions_per_week"]
    minutes = settings["practice_minutes"]

    for subject, mark in marks.items():
        band = mark_band(mark, settings)

        if band == "Needs improvement":
            recommendations.append(make_recommendation(
                rule="weak_mark",
                area=subject,
                priority="high",
                action=(f"Revise the basics of {subject} and do {sessions} short practice "
                        f"sessions (about {minutes} minutes each) this week. Ask your "
                        "teacher which topics cost you the most marks."),
                why=(f"Your current {subject} mark is {mark:g} out of 20, which is below "
                     f"{settings['weak_mark_below']:g}, the limit we use for "
                     "'needs improvement'."),
                goal=f"Aim for at least {settings['pass_mark']:g} in the next marking period.",
            ))
        elif band == "Moderate":
            recommendations.append(make_recommendation(
                rule="moderate_mark",
                area=subject,
                priority="medium",
                action=(f"Keep a steady routine in {subject}: one set of practice "
                        "exercises or one practice test each week, and review the "
                        "questions you got wrong."),
                why=(f"Your current {subject} mark is {mark:g} out of 20. That is above "
                     f"{settings['weak_mark_below']:g} but below "
                     f"{settings['moderate_mark_below']:g}, so there is room to improve."),
                goal=(f"Aim for at least {settings['moderate_mark_below']:g} "
                      "in the next marking period."),
            ))
        elif band == "Strong":
            recommendations.append(make_recommendation(
                rule="strong_mark",
                area=subject,
                priority="info",
                action=f"Keep your current routine in {subject}.",
                why=(f"Your current {subject} mark is {mark:g} out of 20, which is in "
                     f"the strong range (from {settings['strong_mark_from']:g})."),
            ))
    return recommendations


def gap_rule(marks, settings):
    """If one subject is much weaker than another (and not already flagged)."""
    if len(marks) < 2:
        return []

    weakest_subject = None
    weakest_mark = None
    strongest_subject = None
    strongest_mark = None
    for subject, mark in marks.items():
        if weakest_subject is None or mark < weakest_mark:
            weakest_subject = subject
            weakest_mark = mark
        if strongest_subject is None or mark > strongest_mark:
            strongest_subject = subject
            strongest_mark = mark

    gap = strongest_mark - weakest_mark
    already_weak = weakest_mark < settings["weak_mark_below"]
    if gap < settings["big_subject_gap"] or already_weak:
        return []

    return [make_recommendation(
        rule="subject_gap",
        area=weakest_subject,
        priority="medium",
        action=(f"Give {weakest_subject} a few extra study sessions this week, and think "
                f"about what works well for you in {strongest_subject}."),
        why=(f"Your {weakest_subject} mark ({weakest_mark:g}) is {gap:g} points lower than "
             f"your {strongest_subject} mark ({strongest_mark:g}). A big gap can mean one "
             "subject is getting less of your time."),
    )]


def study_time_rule(study_time, has_weak_subject, settings):
    """Suggest more weekly study time when it is very low."""
    if study_time is None or study_time > settings["low_study_time_code"]:
        return []

    code = int(study_time)
    current = STUDY_TIME_LABELS.get(code, "very little time")
    next_level = STUDY_TIME_LABELS.get(code + 1, "a little more time")

    if has_weak_subject:
        priority = "high"
    else:
        priority = "medium"

    return [make_recommendation(
        rule="low_study_time",
        area="Study habits",
        priority=priority,
        action=(f"Plan fixed study times in your week and aim for {next_level} per week "
                "in total. Short sessions on fixed days are easier to keep up than one "
                "long session."),
        why=(f"You reported studying {current} per week. Studying more does not guarantee "
             "higher marks, but very little study time leaves little room to practice "
             "the subjects where you need it."),
    )]


def absence_rules(absences, settings):
    """Flag subjects with many absences."""
    recommendations = []
    for subject, count in absences.items():
        if count >= settings["high_absences"]:
            recommendations.append(make_recommendation(
                rule="high_absences",
                area=subject,
                priority="medium",
                action=(f"Try to attend every {subject} lesson from now on. If something "
                        "makes that hard, talk to your teacher or an adviser early."),
                why=(f"You were absent {count:g} times in {subject}. Many absences make it "
                     "harder to keep up with new topics. We flag "
                     f"{settings['high_absences']:g} or more."),
            ))
    return recommendations


def failure_rules(past_failures, settings):
    """Flag subjects with earlier class failures."""
    recommendations = []
    for subject, count in past_failures.items():
        if count >= settings["failures_flag_from"]:
            recommendations.append(make_recommendation(
                rule="past_failures",
                area=subject,
                priority="medium",
                action=(f"Ask your {subject} teacher which earlier topics you should catch "
                        "up on, and add them to your weekly practice."),
                why=(f"The record shows {count:g} earlier class failure(s) in {subject}. "
                     "Gaps from earlier years can make newer topics harder, so it helps to "
                     "close them early."),
            ))
    return recommendations


def recommend(student, settings, support_threshold=None):
    """Run every rule for one student. Returns a list of recommendations.

    student is a dictionary, for example:
      {"marks": {"Mathematics": 8, "Portuguese": 14},
       "study_time": 1,
       "absences": {"Mathematics": 12},
       "past_failures": {"Mathematics": 1},
       "fail_probability": 0.62}      (optional, from the model)
    """
    marks = student.get("marks", {})
    absences = student.get("absences", {})
    past_failures = student.get("past_failures", {})
    study_time = student.get("study_time")
    fail_probability = student.get("fail_probability")

    has_weak_subject = False
    for mark in marks.values():
        if mark_band(mark, settings) == "Needs improvement":
            has_weak_subject = True

    recommendations = []
    recommendations.extend(model_rule(fail_probability, support_threshold))
    recommendations.extend(subject_rules(marks, settings))
    recommendations.extend(gap_rule(marks, settings))
    recommendations.extend(study_time_rule(study_time, has_weak_subject, settings))
    recommendations.extend(absence_rules(absences, settings))
    recommendations.extend(failure_rules(past_failures, settings))

    if len(recommendations) == 0:
        recommendations.append(make_recommendation(
            rule="no_concerns",
            area="Overall",
            priority="info",
            action="Keep up your current routine.",
            why="The information provided does not trigger any of the rules.",
        ))

    recommendations.sort(key=priority_rank)   # stable: same priority keeps its order
    return recommendations


def explain_student(student, settings, support_threshold=None):
    """Everything the app needs: the subject table, the advice, and the disclaimer."""
    return {
        "subjects": subject_table(student.get("marks", {}), settings),
        "recommendations": recommend(student, settings, support_threshold),
        "disclaimer": DISCLAIMER,
    }


if __name__ == "__main__":
    # A quick demo with an example student
    example = {
        "marks": {"Mathematics": 8, "Portuguese": 14},
        "study_time": 1,
        "absences": {"Mathematics": 12, "Portuguese": 3},
        "past_failures": {"Mathematics": 1, "Portuguese": 0},
        "fail_probability": 0.62,
    }
    result = explain_student(example, load_rule_settings(), support_threshold=0.5)

    print("Subjects:")
    for row in result["subjects"]:
        print("  ", row["subject"], row["mark"], "->", row["performance"])
    print()
    for item in result["recommendations"]:
        print("[" + item["priority"].upper() + "]", item["rule"], "(" + item["area"] + ")")
        print("   Do:  ", item["action"])
        print("   Why: ", item["why"])
        if item["goal"]:
            print("   Goal:", item["goal"])
        print()
    print(result["disclaimer"])