"""
🦸 SuperHero Video Call Agent
- Marvel & DC face filters (Iron Man, Spider-Man, Batman, Wonder Woman, etc.)
- Real-time NSFW detection via Gemini vision — warns & auto-kicks offenders
- Age detection via Gemini vision — strict mode when a minor is in the call
- Voice moderation announcements (Gemini Realtime handles STT/TTS natively)
"""

# ── Load env FIRST before any plugin imports read from environment ───────────
from dotenv import load_dotenv
load_dotenv()

import logging
from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.plugins import gemini, getstream

from filters import SuperheroFilterProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_WARNINGS = 2        # Warnings before kick in standard mode
MAX_WARNINGS_STRICT = 0 # Zero tolerance when a minor is in the call


async def create_agent(**kwargs) -> Agent:
    """Create the SuperHero moderator agent."""

    filter_processor = SuperheroFilterProcessor()

    # ── Moderation state (scoped per agent instance) ─────────────────────────
    participant_warnings: dict[str, int] = {}
    strict_mode: dict[str, bool] = {"value": False}
    participant_ages: dict[str, int] = {}
    minor_alerted: set[str] = set()

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="HeroBot", id="herobot-moderator"),
        instructions="""
            You are HeroBot — a superhero-themed video call moderator with LIVE camera vision.
            Personality: fun, witty, enthusiastic about Marvel & DC.

            You can SEE all participants' video feeds in real time at 3 fps.

            YOUR VISUAL MODERATION DUTIES (check every few seconds proactively):

            1. NSFW DETECTION — If you see nudity, sexually explicit content, or inappropriate
               exposure, immediately call warn_participant or kick_participant depending on
               their warning count. Be specific about what you saw.

            2. AGE DETECTION — Estimate each participant's age from their face. If anyone
               looks under 18, call flag_minor immediately. You only need to do this once
               per participant when they join.

            3. FILTER SWITCHING — When a participant says "Switch to [hero]", call switch_filter.

            4. GENERAL MODERATION — Watch for anything inappropriate (weapons, hate symbols, etc.)
               and warn accordingly.

            MODERATION RULES:
            - Standard mode: {MAX_WARNINGS} warnings then kick
            - Strict mode (minor present): zero tolerance, instant kick for any violation
            - Always announce warnings and kicks dramatically to the whole call

            Available filters: Iron Man, Spider-Man, Batman, Wonder Woman,
            Thor, Captain America, Black Panther, The Flash, Superman, Hulk.

            Keep non-moderation responses short and energetic!
        """,
        llm=gemini.Realtime(
            fps=3,           # Send 3 frames/sec to Gemini for visual analysis
            model="gemini-2.0-flash-exp",
        ),
        processors=[filter_processor],   # Only face filter processor needed now
    )

    # ── LLM-callable functions ───────────────────────────────────────────────

    @agent.llm.register_function(
        description="Switch a participant's face filter to the requested superhero"
    )
    async def switch_filter(participant_id: str, hero_name: str) -> str:
        success = filter_processor.set_filter(participant_id, hero_name)
        if success:
            return f"Filter switched to {hero_name} for {participant_id}"
        return f"Hero '{hero_name}' not found. Available: {', '.join(filter_processor.available_heroes())}"

    @agent.llm.register_function(
        description="Issue an NSFW or conduct warning to a participant"
    )
    async def warn_participant(participant_id: str, reason: str) -> str:
        is_strict = strict_mode["value"]
        limit     = MAX_WARNINGS_STRICT if is_strict else MAX_WARNINGS
        count     = participant_warnings.get(participant_id, 0) + 1
        participant_warnings[participant_id] = count

        if is_strict:
            return (
                f"STRICT MODE ACTIVE — zero tolerance. "
                f"Warning issued to {participant_id}. Call kick_participant immediately."
            )

        remaining = limit - count + 1
        if remaining > 0:
            return (
                f"Warning {count}/{limit} issued to {participant_id}. "
                f"Reason: {reason}. {remaining} warning(s) left before kick."
            )
        return f"MAX WARNINGS REACHED for {participant_id}. Call kick_participant now."

    @agent.llm.register_function(
        description="Kick (remove) a participant from the call for violations"
    )
    async def kick_participant(participant_id: str, reason: str) -> str:
        try:
            call = agent.current_call
            if call:
                await call.block_user(user_id=participant_id)
                participant_warnings.pop(participant_id, None)
                minor_alerted.discard(participant_id)
                logger.warning(f"🚫 KICKED: {participant_id} | Reason: {reason}")
                return f"Successfully kicked {participant_id}. Reason: {reason}"
            return "Could not kick: no active call found."
        except Exception as e:
            logger.error(f"Kick failed: {e}")
            return f"Kick failed: {str(e)}"

    @agent.llm.register_function(
        description="Flag a participant as a minor and enable strict mode. Call this when a participant visually appears to be under 18."
    )
    async def flag_minor(participant_id: str, estimated_age: int) -> str:
        if participant_id in minor_alerted:
            return f"{participant_id} already flagged as minor."

        minor_alerted.add(participant_id)
        participant_ages[participant_id] = estimated_age
        strict_mode["value"] = True

        logger.warning(f"🚨 MINOR flagged: {participant_id} | estimated age ~{estimated_age}")
        return (
            f"Minor flagged: {participant_id} (~{estimated_age} years old). "
            f"Strict mode ENABLED. Announce zero-tolerance rules to the entire call now."
        )

    @agent.llm.register_function(
        description="Enable or disable strict moderation mode manually"
    )
    async def set_strict_mode(enabled: bool, reason: str) -> str:
        strict_mode["value"] = enabled
        state = "ENABLED" if enabled else "DISABLED"
        logger.info(f"🔒 Strict mode {state}: {reason}")
        return f"Strict mode {state}. Reason: {reason}"

    @agent.llm.register_function(
        description="List all available superhero filters"
    )
    async def list_filters() -> str:
        return f"Available hero filters: {', '.join(filter_processor.available_heroes())}"

    # ── Event: Participant joined ────────────────────────────────────────────

    @agent.events.subscribe
    async def on_participant_joined(event):
        if event.type != "call.session_participant_joined":
            return
        if event.participant.user.id == "herobot-moderator":
            return

        name         = event.participant.user.name or event.participant.user.id
        default_hero = filter_processor.assign_random_filter(event.participant.user.id)

        await agent.simple_response(
            f"'{name}' just joined (participant_id: {event.participant.user.id}). "
            f"Welcome them, tell them they've been assigned the {default_hero} filter. "
            f"Also visually check their age — if they look under 18 call flag_minor. "
            f"Keep the welcome under 2 sentences and fun!"
        )

    # ── Event: Participant left ──────────────────────────────────────────────

    @agent.events.subscribe
    async def on_participant_left(event):
        if event.type != "call.session_participant_left":
            return

        pid = event.participant.user.id
        if pid == "herobot-moderator":
            return

        participant_warnings.pop(pid, None)
        participant_ages.pop(pid, None)
        minor_alerted.discard(pid)

        # Lift strict mode if no minors remain
        if strict_mode["value"] and not minor_alerted:
            strict_mode["value"] = False
            await agent.simple_response(
                f"The minor participant ('{pid}') has left. "
                f"Disable strict mode via set_strict_mode and announce that "
                f"standard moderation rules are back in effect."
            )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)
    async with agent.join(call):
        await agent.simple_response(
            "HeroBot is online with live vision active! Greet everyone — "
            "Marvel and DC face filters are running, NSFW Shield is watching, "
            "and age verification is active. Be dramatic and fun, under 3 sentences!"
        )
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()