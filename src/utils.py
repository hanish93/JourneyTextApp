def generate_long_summary(events, signals):
    """
    events:  list of debounced events, e.g. ["drive","passed Tesco Express","turn_left",…]
    signals: list of debounced signals, e.g. [None,"red","red","green",…]
    """
    parts = []
    last_sig = None
    run_drive = 0

    def flush_drive():
        nonlocal run_drive
        if run_drive >= 2:
            parts.append("continued straight for a while")
        run_drive = 0

    for ev, sig in zip(events, signals):
        # 1) signal handling
        if sig == "red" and last_sig != "red":
            flush_drive()
            parts.append("stopped at the red light")
        if sig == "green" and last_sig == "red":
            parts.append("once the signal turned green, I drove on")
        last_sig = sig or last_sig

        # 2) event handling
        if ev.startswith("passed "):
            flush_drive()
            landmark = ev.split(" ",1)[1]
            parts.append(f"passed {landmark}")
        elif ev == "turn_left":
            flush_drive()
            parts.append("turned left")
        elif ev == "turn_right":
            flush_drive()
            parts.append("took a slight right")
        elif ev == "drive":
            run_drive += 1

    # end‐of‐loop cleanup
    flush_drive()

    # make a single sentence
    if not parts:
        return "No notable events detected."

    sent = parts[0].capitalize()
    for p in parts[1:]:
        sent += " and " + p

    # finish punctuation
    if not sent.endswith("."):
        sent += "."

    return sent
