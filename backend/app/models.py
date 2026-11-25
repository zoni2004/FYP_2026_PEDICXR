"""Database tables: studies, results, reports, feedback."""
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Study(Base):
    __tablename__ = "studies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_ref: Mapped[Optional[str]] = mapped_column(String(64))
    age_months: Mapped[int] = mapped_column(Integer)
    sex: Mapped[str] = mapped_column(String(1))
    original_name: Mapped[str] = mapped_column(String(255))
    upload_file: Mapped[str] = mapped_column(String(255))   # file name in uploads/ (as uploaded)
    image_file: Mapped[str] = mapped_column(String(255))    # PNG copy in uploads/ for the viewer
    status: Mapped[str] = mapped_column(String(16), default="review")  # review | approved
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    results: Mapped[list["Result"]] = relationship(back_populates="study", cascade="all, delete-orphan",
                                                   order_by="Result.id")
    report: Mapped[Optional["Report"]] = relationship(back_populates="study", cascade="all, delete-orphan",
                                                      uselist=False)
    feedback: Mapped[list["Feedback"]] = relationship(back_populates="study", cascade="all, delete-orphan",
                                                      order_by="Feedback.id")

    @property
    def code(self) -> str:
        return f"PCX-{self.id:04d}"


class Result(Base):
    __tablename__ = "results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    study_id: Mapped[int] = mapped_column(ForeignKey("studies.id"), index=True)
    model_name: Mapped[str] = mapped_column(String(64))
    finding: Mapped[str] = mapped_column(String(64))
    probability: Mapped[float] = mapped_column(Float)
    heatmap_file: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    study: Mapped[Study] = relationship(back_populates="results")


class Report(Base):
    __tablename__ = "reports"

    study_id: Mapped[int] = mapped_column(ForeignKey("studies.id"), primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="draft")  # draft | approved
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    study: Mapped[Study] = relationship(back_populates="report")


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    study_id: Mapped[int] = mapped_column(ForeignKey("studies.id"), index=True)
    finding: Mapped[str] = mapped_column(String(64))
    agree: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    study: Mapped[Study] = relationship(back_populates="feedback")
