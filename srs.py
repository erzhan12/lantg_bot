from datetime import datetime, timedelta

def sm2_update(easiness: float, interval: int, repetitions: int, quality: int):
    """
    quality: 0..5 (map: correct→4/5, wrong→0..2)
    """
    easiness = max(1.3, easiness + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
    if quality < 3:
        repetitions = 0
        interval = 1
    else:
        repetitions += 1
        if repetitions == 1: interval = 1
        elif repetitions == 2: interval = 6
        else: interval = int(round(interval * easiness))
    return easiness, interval, repetitions

def schedule_next(now: datetime, easiness: float, interval: int, repetitions: int, correct: bool):
    q = 4 if correct else 2
    ef, ivl, reps = sm2_update(easiness or 2.3, interval or 0, repetitions or 0, q)
    next_due = now + timedelta(days=ivl)
    return ef, ivl, reps, next_due

def schedule_initial(now: datetime):
    # Flat/naive for MVP: review after 2 days
    return now + timedelta(days=2), 2, 2.3, 0