from typing import Any, Dict, List, Tuple, Optional
from ..core.schemas import AgentTrace
from .emotion_agent import run_emotion
from .problem_classifier import run_classifier

TIME_BUDGET_OPTIONS = ["under_10_min", "about_30_min", "over_1_hour"]

def _detect_time_budget(text: str) -> Optional[str]:
    t = text.strip().lower()

    if t in TIME_BUDGET_OPTIONS:
        return t

    # light heuristics
    if "under_10" in t or "under 10" in t or "10 min" in t:
        return "under_10_min"
    if "30" in t or "half hour" in t:
        return "about_30_min"
    if "over_1" in t or "over 1" in t or "hour" in t:
        return "over_1_hour"

    return None

def orchestrate(session: Dict[str, Any], user_msg: str) -> Tuple[str, List[AgentTrace]]:
    trace: List[AgentTrace] = []

    session["history"].append({"role": "user", "content": user_msg})

    # Hard rule 1: if time_budget missing, ask ONE question with 3 options then stop
    if session["time_budget"] is None:
        tb = _detect_time_budget(user_msg)
        if tb is None:
            answer = (
                "Before we start, how much time do you want to spend?\n"
                "- under_10_min\n"
                "- about_30_min\n"
                "- over_1_hour\n\n"
                "Reply with one option."
            )
            session["history"].append({"role": "assistant", "content": answer})
            return answer, trace

        session["time_budget"] = tb
        trace.append(AgentTrace(agent="orchestrator", info=f"time_budget set to {tb}"))

        # Hard rule 2: once time_budget known, ask learner for reflection/input then stop
        answer = (
            f"Got it — we’ll do **{tb}**.\n\n"
            "Now share your reflection/input:\n"
            "- what you tried\n"
            "- what went well / not well\n"
            "- what you want to improve"
        )
        session["history"].append({"role": "assistant", "content": answer})
        return answer, trace

    # time_budget known → this message is learner reflection or answers
    session["learner_text"] = user_msg

    # Hard rule 3: ALWAYS run Emotion + Classifier in background after receiving learner input
    emotion_out = run_emotion(user_msg)
    trace.append(AgentTrace(agent="emotion_agent", info="ran"))
    session["tone_json"] = emotion_out

    # Safety check
    if "SAFETY_FLAG" in emotion_out:
        answer = emotion_out.replace("SAFETY_FLAG", "").strip()
        session["history"].append({"role": "assistant", "content": answer})
        return answer, trace

    classifier_out = run_classifier(user_msg)
    trace.append(AgentTrace(agent="problem_classifier", info="ran"))
    session["problem_type_json"] = classifier_out

    # Hard rule 4: respond with brief supportive line + 2–3 tailored follow-up questions
    # (We keep it simple: use emotion line + generic followups for now.)
    supportive_line = emotion_out.strip().split("\n")[0].strip()
    if not supportive_line:
        supportive_line = "Thanks for sharing — I’m with you."

    session["followup_rounds"] += 1

    tb = session["time_budget"]
    if tb == "under_10_min":
        followups = [
            "What’s the *one* thing you want to get out of this in the next 10 minutes?",
            "Which step specifically tripped you up (concept vs. process vs. confidence)?",
        ]
    elif tb == "about_30_min":
        followups = [
            "Where do you think the main bottleneck is (understanding, strategy, or execution)?",
            "What did you try first, and what evidence told you it wasn’t working?",
            "If we change one variable next, what should it be?",
        ]
    else:
        followups = [
            "What’s your target outcome (grade, skill mastery, research insight, or reflection quality)?",
            "What constraints do you have (time, tools, background knowledge)?",
            "Which part should we go deepest on: theory, method, or evaluation?",
        ]

    # 2–3 questions
    followups = followups[:3]

    answer = supportive_line + "\n\n" + "A couple quick questions:\n" + "\n".join(
        [f"{i+1}) {q}" for i, q in enumerate(followups)]
    )

    session["history"].append({"role": "assistant", "content": answer})
    return answer, trace
