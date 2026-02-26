import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import gemini, getstream, ultralytics

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    agent = Agent(
        edge=getstream.Edge(),  # use stream for edge video transport
        agent_user=User(name="AI golf coach"),
        instructions="Read @golf_coach.md",  # read the golf coach markdown instructions
        llm=gemini.Realtime(fps=3),  # Share video with gemini
        # llm=openai.Realtime(fps=3), # use this to switch to openai
        processors=[
            ultralytics.YOLOPoseProcessor(model_path="yolo26n-pose.pt")
        ],  # realtime pose detection with yolo
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    # join the call and open a demo env
    async with agent.join(call):
        # We start a background task or use simple_response with a safety prompt
        await agent.llm.simple_response(
            text="Monitor the video stream. If you see any NSFW content, "
                 "respond with the exact phrase: 'TERMINATE_CALL'."
        )

        # Here is the senior dev trick: listen to the agent's own 'thoughts'
        async for message in agent.llm.listen():
            if "TERMINATE_CALL" in message.text:
                logger.warning(f"NSFW detected! Terminating call {call_id}")
                
                # 1. Send a final warning
                await agent.llm.simple_response(text="NSFW detected. Ending call immediately.")
                
                # 2. Kill the call for everyone
                await call.end_call()
                break 

        await agent.finish()
if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()