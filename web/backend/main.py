from fastapi import FastAPI
from fastapi_socketio import SocketManager

app = FastAPI()
socket_manager = SocketManager(app=app, mount_location="/ws")


@app.get("/")
def getRoot():
    return {"message": "Hello World"}

@app.sio.on('connect')
async def handleNewConnection(sid, *args, **kwargs):
    print("New connection", sid)