from fastapi import FastAPI, APIRouter
from api import all_routers

app = FastAPI()

@app.get("/hello-world")
def hello_world():
    return "hello from FastAPI"

for r in all_routers:
    app.include_router(r)