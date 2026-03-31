from fastapi import FastAPI, Request
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key="sk-f1d48c674fb949888317f85772909496",
    base_url="https://api.deepseek.com"
)


@app.get("/")
def home():
    return {"msg": "hello ai"}


@app.post("/chat")
async def chat(req: Request):
    data = await req.json()
    messages = data["messages"]

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )

    return {"answer": resp.choices[0].message.content}
