from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Dict, List, Optional

class PersonProfileRequest(BaseModel):
    # Identity
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    university: str = Field(..., description="University or institution")
    preferred_career: str = Field(..., description="Preferred career path")

    # Big Five (1-5)
    openness: int = Field(..., ge=1, le=5)
    conscientiousness: int = Field(..., ge=1, le=5)
    extraversion: int = Field(..., ge=1, le=5)
    agreeableness: int = Field(..., ge=1, le=5)
    neuroticism: int = Field(..., ge=1, le=5)

    # Competence (user selects exactly 2)
    dominant_competence: List[str] = Field(default_factory=list)
    # Options: "analytical", "creative", "practical", "people"
    
    # Learning style (user selects exactly 2)
    learning_style: List[str] = Field(default_factory=list)
    # Options: "research_reading", "teamwork_interviewing", 
    #          "hands_on_systems", "brainstorming_ideation"

    # Work values (1-6 rank => we invert later)
    income_importance: int = Field(..., ge=1, le=6)
    impact_importance: int = Field(..., ge=1, le=6)
    stability_importance: int = Field(..., ge=1, le=6)
    variety_importance: int = Field(..., ge=1, le=6)
    recognition_importance: int = Field(..., ge=1, le=6)
    autonomy_importance: int = Field(..., ge=1, le=6)

    # Skills (1-5)
    math: int = Field(..., ge=1, le=5)
    problem_solving: int = Field(..., ge=1, le=5)
    public_speaking: int = Field(..., ge=1, le=5)
    creative: int = Field(..., ge=1, le=5)
    working_with_people: int = Field(..., ge=1, le=5)
    writing: int = Field(..., ge=1, le=5)
    tech_savvy: int = Field(..., ge=1, le=5)
    leadership: int = Field(..., ge=1, le=5)
    networking: int = Field(..., ge=1, le=5)
    negotiation: int = Field(..., ge=1, le=5)
    innovation: int = Field(..., ge=1, le=5)
    programming: int = Field(..., ge=1, le=5)
    languages: int = Field(..., ge=1, le=5)
    empathy: int = Field(..., ge=1, le=5)
    time_management: int = Field(..., ge=1, le=5)
    attention_to_detail: int = Field(..., ge=1, le=5)
    project_management: int = Field(..., ge=1, le=5)
    artistic: int = Field(..., ge=1, le=5)
    research: int = Field(..., ge=1, le=5)
    hands_on_building: int = Field(..., ge=1, le=5)
    teamwork: int = Field(..., ge=1, le=5)
    updates: bool = Field(False, description="Consent for further updates")
    # Interests
    interests: List[str] = Field(
        ..., description="From: investigative, social, artistic, enterprising, realistic, conventional"
    )

class AnalysisResponse(BaseModel):
    success: bool
    message: str
    result: Optional[Dict] = None
    analysis_date: Optional[str] = None

@dataclass
class PersonProfile:
    name: str
    email: str
    university: str
    personality: Dict[str, float]
    work_values: Dict[str, float]
    skills: Dict[str, float]
    interests: List[str]
    preferred_career: str
    dominant_competence: List[str]
    learning_style: List[str]
