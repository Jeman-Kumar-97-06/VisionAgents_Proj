import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from stream_video import StreamVideo
from dotenv import load_dotenv
from getstream import Stream
from vision_agents.core import Agent, AgentLauncher, User, Runner
from vision_agents.plugins import getstream, gemini

app = FastAPI()

load_dotenv()

# 1. Setup CORS so your React app can talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

#create agent:
async def create_agent(**kwargs) -> Agent:
    return Agent(
        edge = getstream.Edge(),
        agent_user = User(name='Assistant', id='agent'),
        instructions = 'Describe what you see be concise',
        llm = gemini.Realtime(fps=3),
    )

async def run_agent_session(call_id: str):
    """The actual logic for joining the call."""
    launcher = AgentLauncher(create_agent=create_agent)
    agent = await launcher.create_agent()
    
    # Connect to the specific room ID sent from frontend [cite: 231]
    call = await agent.create_call("default", call_id) 
    
    async with agent.join(call):
        await agent.simple_response("I'm here! What do you see?")
        await agent.finish()

# Load credentials from .env
API_KEY = os.getenv("STREAM_API_KEY")
API_SECRET = os.getenv("STREAM_API_SEC")
print("Fuck: ",API_KEY)
client = Stream(api_key=API_KEY, api_secret=API_SECRET, timeout=3.0)

# 2. ROUTE: Generate Token for the Frontend User
@app.get("/token/{user_id}")
async def get_token(user_id: str):
    try:
        # Tokens must be generated server-side to keep the secret safe 
        token = client.create_token(user_id=user_id)
        return {"token": token, "apiKey": API_KEY}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3. ROUTE: Start the Vision Agent session
@app.post("/start-call")
async def make_call(data:dict, background_tasks: BackgroundTasks):
    call_id = data.get("call_id")
    if not call_id:
        return {"error":"Missing call_id"},400

    background_tasks.add_task(run_agent_session, call_id)

    return {"status": "Agent is joining the call", "call_id": call_id}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)