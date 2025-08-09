from pydantic import BaseModel

class MoveRequest(BaseModel):
    move: str