from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class IncidentDB(Base):
    __tablename__ = "incidents"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    service: Mapped[str] = mapped_column(String(100))
    severity: Mapped[str] = mapped_column(String(20))
    description: Mapped[str] = mapped_column(Text)