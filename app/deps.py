from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .db import get_sessionmaker


def get_settings():
    return settings


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        yield session
