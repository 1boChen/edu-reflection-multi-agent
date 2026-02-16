import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from openai import OpenAI

ARCHIA_BASE_URL = "https://registry.archia.app/v1"
DEFAULT_MODEL = "priv-claude-sonnet-4-5-20250929"

ORCHESTRATOR_SYSTEM_PROMPT = r"""You are the Orchestrator for a multi-agent STEM reflection assistant.

Your job is to run a multi-turn conversation with the learner and coordinate specialist agents behind the scenes:
- Problem Type Classifier
- Emotion Support Agent
- Recommender Agent
- Expert view Agent (only if time_budget == over_1_hour)
- Summary Agent

You must keep the interaction natural and engaging.
Do NOT output everything in one response.
Instead, proceed in stages across multiple turns.

Conversation state you must track:
- context (optional)
- time_budget: under_10_min | about_30_min | over_1_hour
- learner_text (current)
- problem_type_json
- tone_json
- recommender_json
- expert_json (optional)

Tool-calling / delegation rule:
If you have the ability to call other agents as tools/skills, do so.
Otherwise, you must simulate delegation internally but keep outputs consistent with what those agents would produce.

Hard rules:
1) If time_budget is missing, ask ONE question to collect it with the 3 options. Then stop.
2) Once time_budget is known, ask the learner for their reflection/input if missing. Then stop.
3) After receiving learner input, ALWAYS run Emotion agent + Problem Classifier in the background.
4) You must respond to the learner with:
   - a brief supportive line (based on emotion)
   - 2–3 tailored follow-up questions (based on problem type and time_budget)
   Keep it conversational. Do not show JSON.
5) Only AFTER the learner answers follow-up questions (at least once), run PubMed Recommender and present the search plan in a friendly way (queries + filters + what to look for).
6) If time_budget == over_1_hour, offer: “Want expert perspectives?” If yes, run Expert View agent and present 3 perspectives.
7) At the end, ask: “Do you want a submission-ready summary?” If yes, run Summary Agent and present it.

Response style:
- Friendly, mentor-like, STEM-focused
- Short paragraphs, bullet points when useful
- No developer section labels
- Do not mention “agents” unless the learner asks. You can say “I’ll guide you step by step.”

Safety:
If Emotion agent would flag self-harm/immediate danger, output a brief safety message and stop.

You must always end each turn with a clear question or next step request, unless ending the session.
"""

# --- Specialist prompts (simple placeholders; replace with your tuned prompts later) ---
EMOTION_SYSTEM = """You are an emotion support classifier for a STEM reflection assistant.
Return JSON with:
- emotion: one of [neutral, stressed, confused, frustrated, excited, discouraged]
- support_line: a single short supportive sentence to say to the learner
- safety_flag: true if self-harm/immediate danger is present, else false
Only output JSON."""
CLASSIFIER_SYSTEM = """You are a problem type classifier for STEM reflections.
Return JSON with:
- problem_type: one of [concept_confusion, planning, critique, research_synthesis, debugging, reflection_quality, other]
- rationale: short
Only output JSON."""
RECOMMENDER_SYSTEM = """You are a PubMed search planner.
Given learner context and goal, return JSON with:
- queries: list of 3-6 PubMed-ready query strings
- filters: year range, article types, species, etc. (as text)
- screening_tips: bullet-like list (as strings)
Only output JSON."""
EXPERT_SYSTEM = """You provide 3 expert perspectives on a STEM reflection topic.
Return JSON with:
- perspectives: list of 3 objects {title, gist, how_to_apply}
Only output JSON."""
SUMMARY_SYSTEM = """You write a submission-ready summary of a learner's STEM reflection.
Return JSON with:
- title
- 1_paragraph_summary
- key_points: list of 3-6 bullets
- next_steps: list of 3-5 bullets
Only output JSON."""

def _get_client() -> OpenAI:
    token = os.environ.get("ARCHIA_TOKEN")
    if not token:
        raise RuntimeError("ARCHIA_TOKEN is not set in the environment.")
    return OpenAI(
        base_url=ARCHIA_BASE_URL,
        api_key="not-used",
        default_headers={"Authorization": f"Bearer {token}"},
    )

def call_model_json(client: OpenAI, system: str, user: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """Calls a model and expects it to return a JSON object. Robustness: minimal."""
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = resp.output[0].content[0].text.strip()
    # naive parse; for production add better JSON repair
    import json
    return json.loads(text)

def call_orchestrator_text(client: OpenAI, user: str, state: Dict[str, Any], model: str = DEFAULT_MODEL) -> str:
    """Let orchestrator craft the final user-facing text."""
    # We pass state explicitly so the LLM can stay consistent.
    import json
    prompt = (
        "Current conversation state (JSON):\n"
        f"{json.dumps(state, indent=2)}\n\n"
        "User message:\n"
        f"{user}\n\n"
        "Write the next assistant message to the learner following all hard rules."
    )
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.output[0].content[0].text.strip()

@dataclass
class PrismState:
    context: Optional[str] = None
    time_budget: Optional[str] = None  # under_10_min | about_30_min | over_1_hour
    learner_text: Optional[str] = None

    problem_type_json: Optional[Dict[str, Any]] = None
    tone_json: Optional[Dict[str, Any]] = None
    recommender_json: Optional[Dict[str, Any]] = None
    expert_json: Optional[Dict[str, Any]] = None

    followup_rounds_completed: int = 0
    asked_expert_offer: bool = False
    expert_requested: Optional[bool] = None
    summary_requested: Optional[bool] = None

    history: List[Dict[str, str]] = field(default_factory=list)  # optional logging

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context": self.context,
            "time_budget": self.time_budget,
            "learner_text": self.learner_text,
            "problem_type_json": self.problem_type_json,
            "tone_json": self.tone_json,
            "recommender_json": self.recommender_json,
            "expert_json": self.expert_json,
            "followup_rounds_completed": self.followup_rounds_completed,
            "asked_expert_offer": self.asked_expert_offer,
            "expert_requested": self.expert_requested,
            "summary_requested": self.summary_requested,
        }

class PrismOrchestrator:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.client = _get_client()
        self.model = model
        self.state = PrismState()

    def _is_valid_budget(self, s: str) -> bool:
        return s in {"under_10_min", "about_30_min", "over_1_hour"}

    def _extract_budget_simple(self, msg: str) -> Optional[str]:
        # Simple heuristic; you can replace with a classifier call later
        msg_l = msg.lower().strip()
        mapping = {
            "under_10_min": ["under_10_min", "under 10", "10 min", "ten minutes", "<10"],
            "about_30_min": ["about_30_min", "30", "half hour", "thirty"],
            "over_1_hour":  ["over_1_hour", "over 1", "1 hour", "60", "more than an hour"],
        }
        for k, keys in mapping.items():
            if any(x in msg_l for x in keys):
                return k
        return None

    def _ask_time_budget(self) -> str:
        return (
            "How much time do you want to spend on this reflection right now?\n"
            "- under_10_min\n"
            "- about_30_min\n"
            "- over_1_hour\n\n"
            "Just reply with one option."
        )

    def _ask_for_reflection(self) -> str:
        return (
            "Got it. Paste your STEM reflection (or a rough draft) and tell me what it’s about.\n"
            "If you don’t have a draft yet, share: topic + what you’re trying to argue/learn + what feels hardest."
        )

    def _format_recommender_plan(self) -> str:
        r = self.state.recommender_json or {}
        queries = r.get("queries", [])
        filters = r.get("filters", "")
        tips = r.get("screening_tips", [])

        lines = ["Here’s a PubMed search plan we can use:"]
        if queries:
            lines.append("\nQueries to try:")
            for q in queries:
                lines.append(f"- {q}")
        if filters:
            lines.append(f"\nHelpful filters:\n- {filters}")
        if tips:
            lines.append("\nWhat to look for when screening:")
            for t in tips:
                lines.append(f"- {t}")
        lines.append("\nWant me to tailor the queries to a specific subtopic or population (e.g., K-12, undergrads, clinical STEM, etc.)?")
        return "\n".join(lines)

    def _format_expert_views(self) -> str:
        e = self.state.expert_json or {}
        perspectives = e.get("perspectives", [])
        if not perspectives:
            return "I couldn’t generate expert perspectives this time. Want to try again with a bit more topic detail?"

        lines = ["Here are three expert perspectives you can borrow (and cite/echo) in your reflection:"]
        for p in perspectives[:3]:
            lines.append(f"\n- **{p.get('title','Perspective')}**: {p.get('gist','')}\n  How to apply: {p.get('how_to_apply','')}")
        lines.append("\nDo you want a submission-ready summary now?")
        return "\n".join(lines)

    def _format_summary(self) -> str:
        s = self.state.recommender_json  # not correct; will be set by summary agent
        s = self.state  # placeholder

    def step(self, user_msg: str) -> str:
        # Log
        self.state.history.append({"role": "user", "content": user_msg})

        # Rule 1: time_budget missing → ask and stop (enforced in code)
        if not self.state.time_budget:
            maybe = self._extract_budget_simple(user_msg)
            if maybe and self._is_valid_budget(maybe):
                self.state.time_budget = maybe
            else:
                return self._ask_time_budget()

        # Rule 2: time_budget known, but learner_text missing → ask and stop
        if not self.state.learner_text:
            # If the message looks like a reflection, accept it
            # (Very simple heuristic: length)
            if len(user_msg.strip()) >= 40 and self._extract_budget_simple(user_msg) is None:
                self.state.learner_text = user_msg.strip()
            else:
                return self._ask_for_reflection()

        # We have learner_text; run Emotion + Classifier every turn after learner input (Rule 3)
        # For follow-up rounds, we treat the new user message as additional info appended
        if user_msg.strip() and user_msg.strip() != self.state.learner_text:
            # append as additional learner input
            self.state.learner_text += "\n\n[Additional learner input]\n" + user_msg.strip()

        # Background calls:
        tone = call_model_json(
            self.client,
            EMOTION_SYSTEM,
            f"Learner text:\n{self.state.learner_text}\n\nReturn JSON only.",
            model=self.model,
        )
        self.state.tone_json = tone

        if tone.get("safety_flag") is True:
            # Safety rule
            return (
                "I’m really sorry you’re going through this. If you’re in immediate danger or might hurt yourself, "
                "please call 911 (US) or your local emergency number right now.\n"
                "If you can, reach out to someone you trust or contact the 988 Suicide & Crisis Lifeline (US) by calling or texting 988.\n"
                "If you tell me what country you’re in, I can share the right crisis contact options."
            )

        cls = call_model_json(
            self.client,
            CLASSIFIER_SYSTEM,
            f"Learner text:\n{self.state.learner_text}\n\nReturn JSON only.",
            model=self.model,
        )
        self.state.problem_type_json = cls

        # Decide what stage we’re in:
        # If we haven't done at least 1 follow-up round, produce supportive line + 2–3 questions (Rule 4)
        if self.state.followup_rounds_completed == 0:
            # Let orchestrator craft the actual text (it will see tone_json/problem_type_json/time_budget)
            assistant_text = call_orchestrator_text(self.client, user_msg, self.state.to_dict(), model=self.model)
            self.state.history.append({"role": "assistant", "content": assistant_text})
            self.state.followup_rounds_completed = 1
            return assistant_text

        # After at least one follow-up answer, run recommender (Rule 5)
        if not self.state.recommender_json:
            rec = call_model_json(
                self.client,
                RECOMMENDER_SYSTEM,
                f"Time budget: {self.state.time_budget}\n\nLearner text:\n{self.state.learner_text}\n\nReturn JSON only.",
                model=self.model,
            )
            self.state.recommender_json = rec
            return self._format_recommender_plan()

        # If over_1_hour, offer expert perspectives (Rule 6)
        if self.state.time_budget == "over_1_hour" and not self.state.asked_expert_offer:
            self.state.asked_expert_offer = True
            return "Want expert perspectives? If yes, reply: yes. If not, reply: no."

        # If user said yes to experts
        if self.state.time_budget == "over_1_hour" and self.state.expert_requested is None:
            if user_msg.strip().lower() in {"yes", "y"}:
                self.state.expert_requested = True
                ex = call_model_json(
                    self.client,
                    EXPERT_SYSTEM,
                    f"Learner text:\n{self.state.learner_text}\n\nReturn JSON only.",
                    model=self.model,
                )
                self.state.expert_json = ex
                return self._format_expert_views()
            elif user_msg.strip().lower() in {"no", "n"}:
                self.state.expert_requested = False
                return "No problem. Do you want a submission-ready summary?"

        # Ask for summary if not asked yet (Rule 7)
        if self.state.summary_requested is None:
            if user_msg.strip().lower() in {"yes", "y"}:
                self.state.summary_requested = True
                summ = call_model_json(
                    self.client,
                    SUMMARY_SYSTEM,
                    f"Time budget: {self.state.time_budget}\n\nLearner text:\n{self.state.learner_text}\n\nReturn JSON only.",
                    model=self.model,
                )
                # Render it nicely
                title = summ.get("title", "Reflection Summary")
                para = summ.get("1_paragraph_summary", "")
                key_points = summ.get("key_points", [])
                next_steps = summ.get("next_steps", [])

                lines = [f"**{title}**", "", para]
                if key_points:
                    lines += ["", "Key points:"]
                    lines += [f"- {kp}" for kp in key_points]
                if next_steps:
                    lines += ["", "Next steps:"]
                    lines += [f"- {ns}" for ns in next_steps]
                return "\n".join(lines)
            else:
                return "Do you want a submission-ready summary? (yes/no)"

        # Default: keep coaching
        return "Want to refine the summary, or tighten it to match a specific rubric (e.g., IMRaD-style, claim–evidence–reasoning, or reflection prompts)?"

def main():
    bot = PrismOrchestrator(model=DEFAULT_MODEL)
    print("Prism Orchestrator (type 'exit' to quit)\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        reply = bot.step(user)
        print(f"\nAssistant: {reply}\n")

if __name__ == "__main__":
    main()
