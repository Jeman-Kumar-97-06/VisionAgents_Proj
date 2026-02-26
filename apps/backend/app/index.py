import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from stream_chat import StreamChat
from vision_agents.plugins import gemini, getstream, ultralytics
from fastapi.middleware.gzip import GZipMiddleware

import os 

logger = logging.getLogger(__name__)

load_dotenv()

server_client = StreamChat(
    api_key=os.getenv("STREAM_API_KEY"), 
    api_secret=os.getenv("STREAM_API_SECRET")
)

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

    async with agent.join(call):
        await agent.llm.simple_response(
            text="Monitor all participants. If you see NSFW content, identify the participant "
                 "and respond with: 'KICK [participant_id]'."
        )

        async for event in agent.llm.listen():
            # Listen for the specific kick command from the LLM
            if "KICK" in event.text:
                # Extract the ID (e.g., 'KICK user-123' -> 'user-123')
                target_id = event.text.split(" ")[1]
                
                logger.warning(f"Violation detected! Removing user: {target_id}")
                
                # Use the Stream call object to kick the specific user
                await call.remove_members([target_id]) 
                
                # Optional: The agent can explain why it happened
                await agent.llm.simple_response(text=f"User {target_id} removed for safety violations.")

        await agent.finish()

runner = Runner(AgentLauncher(create_agent=create_agent, join_call=join_call))

# Adding a custom endpoint
@runner.fast_api.get("/token/{user_id}")
async def get_token(user_id: str):
    """Generates a Stream JWT for the React frontend[cite: 245, 252]."""
    token = server_client.create_token(user_id)
    return {"token": token}

runner.fast_api.add_middleware(GZipMiddleware, minimum_size=1000)

if __name__ == "__main__":
    runner.cli()