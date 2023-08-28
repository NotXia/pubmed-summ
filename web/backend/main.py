import sys
from os import path
sys.path.append(path.abspath("../../framework"))
from fastapi import FastAPI
import socketio
from summarize import ClustererSummarizer
from modules.Document import Document
from modules.Cluster import Cluster
from fastapi.concurrency import run_in_threadpool
from threading import Lock
import asyncio

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = FastAPI()
sio_asgi = socketio.ASGIApp(socketio_server=sio)
app.mount("/ws", sio_asgi)



@sio.on("connect")
async def handleNewConnection(sid, *args, **kwargs):
    print("New connection", sid)


curr_requests_lock = Lock()
curr_requests = []


def startSummarizer(loop, request_id: int, query: str):
    with curr_requests_lock:
        if request_id not in curr_requests:
            curr_requests.append(request_id)
        else:
            return

    def onClustersCreated(clusters: list[Cluster]):
        loop.create_task(sio.emit("on_clusters_created", [c.toJSON() for c in clusters], to=request_id))

    def onDocumentSummary(document: Document):
        loop.create_task(sio.emit("on_document_ready", document.toJSON(), to=request_id))

    def onClusterSummary(cluster: Cluster):
        loop.create_task(sio.emit("on_cluster_ready", cluster.toJSON(), to=request_id))

    summarizer = ClustererSummarizer(fs_cache=True)
    summarizer(
        query = query,
        max_fetched = 30,
        onClustersCreated = onClustersCreated,
        onDocumentSummary = onDocumentSummary,
        onClusterSummary = onClusterSummary
    )

    with curr_requests_lock:
        curr_requests.remove(request_id)

@sio.on("query")
async def handleQuery(sid, query):
    request_id = hash(query)
    sio.enter_room(sid, request_id)
    await sio.emit("on_query_received", data={"id": request_id})

    await run_in_threadpool(startSummarizer, loop=asyncio.get_event_loop(), request_id=request_id, query=query)
