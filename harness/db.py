import os
from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    Enum,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# --- Database Connection ---
# Uses DATABASE_URL if set (e.g. Railway PostgreSQL), otherwise falls back to local SQLite.
_default_db = "sqlite:///" + os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment.db"
)
DATABASE_URL = os.environ.get("DATABASE_URL", _default_db)

# create_engine creates a connection pool to the database.
# echo=False means it won't print every SQL query (set True for debugging).
engine = create_engine(DATABASE_URL, echo=False)

# sessionmaker creates a factory for database sessions.
# A session is like a "workspace" — you add objects, then commit them all at once.
SessionLocal = sessionmaker(bind=engine)

# declarative_base returns a base class that our models inherit from.
# It connects Python classes to database tables.
Base = declarative_base()


# --- Models ---
# Each class below maps to one database table.
# Column() defines a column. The first argument is the type.
# primary_key=True means it auto-increments and uniquely identifies each row.
# ForeignKey links one table to another.
# nullable=False means the field is required.
# nullable=True (default) means the field is optional.


class Paragraph(Base):
    """Stores the Wikipedia source paragraphs used in the experiment."""

    __tablename__ = "paragraphs"

    id = Column(Integer, primary_key=True)
    url = Column(String, nullable=False)           # Wikipedia URL
    category = Column(String, nullable=False)       # e.g. "science", "history", "news"
    original_text = Column(Text, nullable=False)    # the actual paragraph text
    corrupted_text = Column(Text, nullable=True)    # version with injected false sentences (if applicable)

    # relationship() lets you access related trials via paragraph.trials
    trials = relationship("Trial", back_populates="paragraph")


class Trial(Base):
    """One row per experimental trial. Links a paragraph to a set of conditions."""

    __tablename__ = "trials"

    id = Column(Integer, primary_key=True)
    paragraph_id = Column(Integer, ForeignKey("paragraphs.id"), nullable=False)

    # The three independent variables:
    trust_level = Column(
        Enum("blind", "medium_skeptic", "full_skeptic", name="trust_level_enum"),
        nullable=False,
    )
    deception_level = Column(
        Enum("truthful", "medium", "full_hallucination", "none", name="deception_level_enum"),
        nullable=False,
    )  # "none" is for Series 2 where there's no deception module
    attack_type = Column(
        Enum("none", "env_injection", "model_tampering", name="attack_type_enum"),
        nullable=False,
    )

    series = Column(Integer, nullable=False)  # 1, 2, or 3
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    paragraph = relationship("Paragraph", back_populates="trials")
    messages = relationship("Message", back_populates="trial", order_by="Message.turn_number")
    result = relationship("Result", back_populates="trial", uselist=False)
    # uselist=False because each trial has exactly one result


class Message(Base):
    """Every message exchanged in a trial — system prompts, agent responses, follow-ups."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey("trials.id"), nullable=False)

    agent = Column(
        Enum("agent_a", "agent_b", "judge", "harness", name="agent_enum"),
        nullable=False,
    )  # who sent this message. "harness" is for system prompts.
    role = Column(
        Enum("system", "user", "assistant", name="role_enum"),
        nullable=False,
    )  # maps to Claude API roles: system prompt, user input, assistant response
    content = Column(Text, nullable=False)       # the actual message text
    turn_number = Column(Integer, nullable=False) # ordering within the trial (0, 1, 2, ...)
    created_at = Column(DateTime, default=datetime.utcnow)

    trial = relationship("Trial", back_populates="messages")


class Result(Base):
    """The scored outcome of each trial."""

    __tablename__ = "results"

    id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey("trials.id"), nullable=False)

    # Agent A's output
    agent_a_decision = Column(
        Enum("accept", "reject", name="decision_enum"),
        nullable=False,
    )
    agent_a_confidence = Column(Integer, nullable=False)  # 1-5
    agent_a_reasoning = Column(Text, nullable=False)

    # Judge's output
    judge_label = Column(
        Enum("truthful", "medium_deception", "full_hallucination", name="judge_label_enum"),
        nullable=False,
    )
    judge_justification = Column(Text, nullable=False)
    judge_comprehension = Column(
        Enum("accurate", "partially_accurate", "inaccurate", name="comprehension_enum"),
        nullable=False,
    )

    # Final score
    winner = Column(
        Enum("agent_a", "agent_b", name="winner_enum"),
        nullable=False,
    )
    score_reasoning = Column(Text, nullable=False)

    trial = relationship("Trial", back_populates="result")


# --- Helper Functions ---


def init_db():
    """Creates all tables in the database if they don't exist yet."""
    Base.metadata.create_all(engine)


def get_session():
    """Returns a new database session. Use in a with block:

    with get_session() as session:
        session.add(something)
        session.commit()
    """
    return SessionLocal()


def save_paragraph(session, url, category, original_text, corrupted_text=None):
    """Saves a Wikipedia paragraph to the database. Returns the Paragraph object."""
    paragraph = Paragraph(
        url=url,
        category=category,
        original_text=original_text,
        corrupted_text=corrupted_text,
    )
    session.add(paragraph)
    session.flush()  # flush assigns the auto-generated id without committing
    return paragraph


def save_trial(session, paragraph_id, trust_level, deception_level, attack_type, series):
    """Creates a new trial record. Returns the Trial object."""
    trial = Trial(
        paragraph_id=paragraph_id,
        trust_level=trust_level,
        deception_level=deception_level,
        attack_type=attack_type,
        series=series,
    )
    session.add(trial)
    session.flush()
    return trial


def save_message(session, trial_id, agent, role, content, turn_number):
    """Logs a single message in a trial's conversation."""
    message = Message(
        trial_id=trial_id,
        agent=agent,
        role=role,
        content=content,
        turn_number=turn_number,
    )
    session.add(message)
    session.flush()
    return message


def save_result(session, trial_id, agent_a_decision, agent_a_confidence,
                agent_a_reasoning, judge_label, judge_justification,
                judge_comprehension, winner, score_reasoning):
    """Saves the scored result of a trial."""
    result = Result(
        trial_id=trial_id,
        agent_a_decision=agent_a_decision,
        agent_a_confidence=agent_a_confidence,
        agent_a_reasoning=agent_a_reasoning,
        judge_label=judge_label,
        judge_justification=judge_justification,
        judge_comprehension=judge_comprehension,
        winner=winner,
        score_reasoning=score_reasoning,
    )
    session.add(result)
    session.flush()
    return result
