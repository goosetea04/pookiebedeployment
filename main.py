import firebase_admin
from firebase_admin import credentials, firestore, db
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json
import io
import uvicorn
from dataclasses import dataclass, asdict
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import openai
import os
from openai import OpenAI
import careers 

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Pydantic models for request/response (keeping existing ones)
class PersonProfileRequest(BaseModel):
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    university: str = Field(..., description="University or institution")
    preferred_career: str = Field(..., description="Preferred career path")
    
    # Personality traits (1-5 scale)
    openness: int = Field(..., ge=1, le=5, description="Openness to experience")
    conscientiousness: int = Field(..., ge=1, le=5, description="Conscientiousness")
    extraversion: int = Field(..., ge=1, le=5, description="Extraversion")
    agreeableness: int = Field(..., ge=1, le=5, description="Agreeableness")
    neuroticism: int = Field(..., ge=1, le=5, description="Neuroticism")
    
    # Work values (1-6 ranking, 1=most important)
    income_importance: int = Field(..., ge=1, le=6, description="Importance of income (1=most important, 6=least)")
    impact_importance: int = Field(..., ge=1, le=6, description="Importance of making an impact")
    stability_importance: int = Field(..., ge=1, le=6, description="Importance of job stability")
    variety_importance: int = Field(..., ge=1, le=6, description="Importance of variety in work")
    recognition_importance: int = Field(..., ge=1, le=6, description="Importance of recognition")
    autonomy_importance: int = Field(..., ge=1, le=6, description="Importance of autonomy")
    
    # Skills (1-5 scale)
    math: int = Field(..., ge=1, le=5, description="Math skills")
    problem_solving: int = Field(..., ge=1, le=5, description="Problem solving skills")
    public_speaking: int = Field(..., ge=1, le=5, description="Public speaking skills")
    creative: int = Field(..., ge=1, le=5, description="Creative skills")
    working_with_people: int = Field(..., ge=1, le=5, description="People skills")
    writing: int = Field(..., ge=1, le=5, description="Writing & communication")
    tech_savvy: int = Field(..., ge=1, le=5, description="Technology skills")
    leadership: int = Field(..., ge=1, le=5, description="Leadership & management")
    networking: int = Field(..., ge=1, le=5, description="Networking & relationship building")
    programming: int = Field(..., ge=1, le=5, description="Programming proficiency")
    empathy: int = Field(..., ge=1, le=5, description="Empathy")
    time_management: int = Field(..., ge=1, le=5, description="Time management")
    attention_to_detail: int = Field(..., ge=1, le=5, description="Attention to detail")
    project_management: int = Field(..., ge=1, le=5, description="Project management")
    research: int = Field(..., ge=1, le=5, description="Research skills")
    teamwork: int = Field(..., ge=1, le=5, description="Teamwork")
    # Consent for further updates 
    updates: bool =  Field(..., description="Consent for further updates")
    # Interests (multiple selection)
    interests: List[str] = Field(..., description="List of interests from: investigative, social, artistic, enterprising, realistic, conventional")

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

class AICareerMatcher:
    def __init__(self):
        # Enhanced O*NET Job Database with similar roles mapping
        self.onet_jobs = careers.onet_jobs

    def create_profile_from_request(self, request: PersonProfileRequest) -> PersonProfile:
        """Create PersonProfile from API request"""
        personality = {
            "openness": request.openness,
            "conscientiousness": request.conscientiousness,
            "extraversion": request.extraversion,
            "agreeableness": request.agreeableness,
            "neuroticism": request.neuroticism
        }
        
        work_values = {
            "income": 7 - request.income_importance,
            "impact": 7 - request.impact_importance,
            "stability": 7 - request.stability_importance,
            "variety": 7 - request.variety_importance,
            "recognition": 7 - request.recognition_importance,
            "autonomy": 7 - request.autonomy_importance
        }
        
        skills = {
            "math": request.math,
            "problem_solving": request.problem_solving,
            "public_speaking": request.public_speaking,
            "creative": request.creative,
            "working_with_people": request.working_with_people,
            "writing": request.writing,
            "tech_savvy": request.tech_savvy,
            "leadership": request.leadership,
            "networking": request.networking,
            "programming": request.programming,
            "empathy": request.empathy,
            "time_management": request.time_management,
            "attention_to_detail": request.attention_to_detail,
            "project_management": request.project_management,
            "research": request.research,
            "teamwork": request.teamwork
        }
        
        return PersonProfile(
            name=request.name,
            email=request.email,
            university=request.university,
            personality=personality,
            work_values=work_values,
            skills=skills,
            interests=request.interests,
            preferred_career=request.preferred_career
        )

    def get_top_job_matches(self, profile: PersonProfile, top_n: int = 3) -> List[str]:
        """
        Calculate match scores for all jobs and return top N job names
        This is a lightweight calculation without AI insights
        """
        job_scores = []
        
        for job_name in self.onet_jobs.keys():
            job = self.onet_jobs[job_name]
            
            # Quick scoring calculation
            skills_score = self._calculate_skills_match(profile.skills, job["skills"])
            values_score = self._calculate_values_match(profile.work_values, job["work_values"])
            interests_score = self._calculate_interests_match(profile.interests, job["interests"])
            work_styles_score = self._calculate_work_styles_match(profile.personality, job.get("work_styles", {}))
            
            overall_score = (skills_score * 0.3 + values_score * 0.25 + 
                           interests_score * 0.2 + work_styles_score * 0.25)
            
            job_scores.append((job_name, overall_score))
        
        # Sort by score and return top N job names
        job_scores.sort(key=lambda x: x[1], reverse=True)
        return [job_name for job_name, _ in job_scores[:top_n]]

    async def generate_ai_insights(self, profile: PersonProfile, job_name: str, match_data: Dict) -> Dict:
        """Generate AI-powered insights for a specific job match"""
        job = self.onet_jobs[job_name]
        
        # Prepare user data summary for AI
        user_summary = self._prepare_user_summary(profile, match_data)
        
        # Generate AI summary
        ai_summary = await self._generate_ai_summary(profile, job_name, job, match_data)
        
        # Generate top keywords and O-NET categories
        keywords = await self._generate_keywords(job_name, job)
        onet_categories = await self._generate_onet_categories(job_name, job)
        
        # Generate action plan
        action_plan = await self._generate_action_plan(profile, job_name, match_data)
        
        # Generate career story
        career_story = await self._generate_career_story(profile, job_name, match_data)
        
        # Generate interview insights
        interview_insights = await self._generate_interview_insights(profile, job_name, match_data)
        
        return {
            "ai_summary": ai_summary,
            "keywords": keywords,
            "onet_categories": onet_categories,
            "action_plan": action_plan,
            "career_story": career_story,
            "interview_insights": interview_insights,
            "similar_roles": job.get("similar_roles", [])
        }

    def _prepare_user_summary(self, profile: PersonProfile, match_data: Dict) -> str:
        """Prepare a concise summary of user data for AI processing"""
        top_skills = sorted(profile.skills.items(), key=lambda x: x[1], reverse=True)[:5]
        top_values = sorted(profile.work_values.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return f"""
        User: {profile.name}
        Top Skills: {', '.join([f'{skill}: {level}/5' for skill, level in top_skills])}
        Top Work Values: {', '.join([f'{value}: {level}/6' for value, level in top_values])}
        Interests: {', '.join(profile.interests)}
        Overall Match: {match_data['overall_match']}%
        Strengths: {len(match_data['strengths'])} identified
        Improvement Areas: {len(match_data['improvements'])} identified
        """

    async def _generate_ai_summary(self, profile: PersonProfile, job_name: str, job: Dict, match_data: Dict) -> str:
        """Generate AI summary of job fit"""
        prompt = f"""
        Create a personalized 2-3 paragraph summary for {profile.name} regarding their fit for the {job_name} position.
        
        Key Information:
        - Overall Match: {match_data['overall_match']}%
        - Skills Match: {match_data['breakdown']['skills_match']}%
        - Values Match: {match_data['breakdown']['values_match']}%
        - Interests Match: {match_data['breakdown']['interests_match']}%
        - Work Styles Match: {match_data['breakdown']['work_styles_match']}%
        
        User Strengths: {', '.join(match_data['strengths'])}
        Areas for Improvement: {len(match_data['improvements'])} key areas identified
        
        Write in a encouraging, professional tone that:
        1. Acknowledges their strengths and natural fit
        2. Addresses any gaps constructively
        3. Provides realistic outlook on their candidacy
        4. Motivates action toward career goals
        
        Keep it concise but insightful.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"AI analysis temporarily unavailable. Based on your {match_data['overall_match']}% match score, you show strong potential for this role with {len(match_data['strengths'])} key strengths identified."

    async def _generate_keywords(self, job_name: str, job: Dict) -> List[str]:
        """Generate 4 relevant keywords for the job"""
        if "job_keywords" in job:
            return job["job_keywords"][:4]
        
        prompt = f"""
        Generate exactly 4 keywords that best represent the {job_name} role.
        These should be:
        - Single words or short phrases (2-3 words max)
        - Industry-relevant
        - What someone would associate with this career
        - Professional and specific
        
        Return only the keywords, separated by commas.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.5
            )
            keywords = [k.strip() for k in response.choices[0].message.content.strip().split(',')]
            return keywords[:4]
        except Exception:
            return ["professional", "skilled", "dedicated", "growth-oriented"]

    async def _generate_onet_categories(self, job_name: str, job: Dict) -> Dict[str, List[str]]:
        """Generate 3 words for each O-NET category"""
        categories = {
            "skills": list(job["skills"].keys())[:3],
            "work_values": list(job["work_values"].keys())[:3],
            "work_styles": list(job.get("work_styles", {}).keys())[:3],
            "interests": list(job["interests"].keys())[:3]
        }
        
        # Clean up the categories
        for category, items in categories.items():
            categories[category] = [item.replace('_', ' ').title() for item in items]
        
        return categories

    async def _generate_action_plan(self, profile: PersonProfile, job_name: str, match_data: Dict) -> Dict:
        """Generate personalized action plan"""
        improvements = match_data.get('improvements', [])
        
        prompt = f"""
        Create a focused action plan for {profile.name} to become a competitive candidate for {job_name}.
        
        Current gaps to address:
        {chr(10).join([f"- {imp['skill']}: Gap of {imp['required_level'] - imp['current_level']:.1f} points" for imp in improvements[:3]])}
        
        Provide:
        1. Top Needs (3 priority areas)
        2. Action Items (4-5 specific, actionable steps)
        
        Make it practical and achievable within 3-6 months.
        Format as JSON with "top_needs" and "action_items" arrays.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.6
            )
            
            content = response.choices[0].message.content.strip()
            # Try to extract JSON, fallback if needed
            try:
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass
            
            # Fallback structure
            return {
                "top_needs": [
                    f"Develop {improvements[0]['skill']} skills" if improvements else "Enhance core competencies",
                    "Build relevant experience",
                    "Network in target industry"
                ],
                "action_items": [
                    "Take online courses in key skill areas",
                    "Seek relevant projects or volunteer opportunities", 
                    "Connect with professionals in the field",
                    "Practice interviewing for this role type",
                    "Build a portfolio showcasing relevant work"
                ]
            }
            
        except Exception:
            return {
                "top_needs": ["Skill development", "Experience building", "Industry networking"],
                "action_items": ["Complete relevant courses", "Gain hands-on experience", "Build professional network", "Prepare interview materials"]
            }

    async def _generate_career_story(self, profile: PersonProfile, job_name: str, match_data: Dict) -> str:
        """Generate a compelling career narrative"""
        strengths = match_data.get('strengths', [])
        
        prompt = f"""
        Write a compelling 2-paragraph career story for {profile.name} pursuing {job_name}.
        
        Their strengths: {', '.join(strengths[:4])}
        Match score: {match_data['overall_match']}%
        
        Create a narrative that:
        1. First paragraph: Connects their background and strengths to this career path naturally
        2. Second paragraph: Shows vision for their future growth and impact in this role
        
        Write in first person as if they're telling their story.
        Be authentic, confident, and forward-looking.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return f"My journey toward {job_name} has been shaped by my natural strengths and genuine passion for this field. Through my experiences, I've developed key capabilities that align well with what this role demands. I'm excited about the opportunity to bring my skills and perspective to make a meaningful impact in this position and grow alongside the organization."

    async def _generate_interview_insights(self, profile: PersonProfile, job_name: str, match_data: Dict) -> Dict:
        """Generate interview-ready insights"""
        strengths = match_data.get('strengths', [])
        
        prompt = f"""
        Create interview preparation insights for {profile.name} applying for {job_name}.
        
        Their key strengths: {', '.join(strengths[:4])}
        
        Provide:
        1. Key Selling Points (3 main strengths to emphasize)
        2. Story Examples (3 specific scenarios they should prepare)
        3. Questions to Ask (3 thoughtful questions for the interviewer)
        
        Format as JSON with these three arrays.
        Make it specific and actionable for interview success.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
                temperature=0.6
            )
            
            content = response.choices[0].message.content.strip()
            try:
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass
            
            # Fallback
            return {
                "key_selling_points": [
                    "Strong analytical and problem-solving abilities",
                    "Excellent communication and interpersonal skills", 
                    "Proven ability to adapt and learn quickly"
                ],
                "story_examples": [
                    "Time you solved a complex problem using analytical thinking",
                    "Situation where you collaborated effectively with a team",
                    "Example of learning a new skill quickly and applying it successfully"
                ],
                "questions_to_ask": [
                    "What does success look like in this role after the first year?",
                    "What are the biggest challenges facing the team right now?",
                    "How does this role contribute to the organization's strategic goals?"
                ]
            }
            
        except Exception:
            return {
                "key_selling_points": ["Strong foundational skills", "Growth mindset", "Team collaboration"],
                "story_examples": ["Problem-solving example", "Teamwork scenario", "Learning achievement"],
                "questions_to_ask": ["Role expectations", "Team challenges", "Growth opportunities"]
            }

    def calculate_job_match(self, profile: PersonProfile, job_name: str) -> Dict:
        """Calculate comprehensive match percentage and details for a specific job"""
        job = self.onet_jobs[job_name]
        
        # Skills match (30% weight)
        skills_score = self._calculate_skills_match(profile.skills, job["skills"])
        
        # Work values match (25% weight) 
        values_score = self._calculate_values_match(profile.work_values, job["work_values"])
        
        # Interests match (20% weight)
        interests_score = self._calculate_interests_match(profile.interests, job["interests"])
        
        # Work styles match (25% weight)
        work_styles_score = self._calculate_work_styles_match(profile.personality, job.get("work_styles", {}))
        
        # Overall match
        overall_match = (skills_score * 0.3 + values_score * 0.25 + 
                        interests_score * 0.2 + work_styles_score * 0.25) * 100
        
        # Identify strengths and improvement areas
        strengths, improvements = self._identify_strengths_improvements(profile, job)
        
        return {
            "job_name": job_name,
            "onet_code": job["onet_code"],
            "overall_match": round(overall_match, 1),
            "breakdown": {
                "skills_match": round(skills_score * 100, 1),
                "values_match": round(values_score * 100, 1),
                "interests_match": round(interests_score * 100, 1),
                "work_styles_match": round(work_styles_score * 100, 1)
            },
            "strengths": strengths,
            "improvements": improvements,
            "required_skills": job["required_skills"]
        }

    def _calculate_skills_match(self, user_skills: Dict, job_skills: Dict) -> float:
        """Calculate skills compatibility score"""
        scores = []
        
        for skill, importance in job_skills.items():
            user_level = user_skills.get(skill, 2.5)
            
            # Normalize both to 0-1 scale
            normalized_importance = importance / 5.0
            normalized_user = user_level / 5.0
            
            # Calculate match
            if normalized_user >= normalized_importance:
                score = 1.0
            else:
                gap = normalized_importance - normalized_user
                score = max(0, 1 - (gap * 2))
            
            scores.append(score * normalized_importance)
        
        return sum(scores) / sum(job_skills.values()) * 5 if scores else 0

    def _calculate_values_match(self, user_values: Dict, job_values: Dict) -> float:
        """Calculate work values compatibility"""
        scores = []
        
        for value, job_importance in job_values.items():
            user_importance = user_values.get(value, 3.5)
            
            # Normalize to 0-1
            norm_job = job_importance / 5.0
            norm_user = user_importance / 6.0
            
            # Calculate similarity
            difference = abs(norm_job - norm_user)
            similarity = 1 - difference
            
            scores.append(similarity * norm_job)
        
        return sum(scores) / sum(job_values.values()) * 5 if scores else 0

    def _calculate_interests_match(self, user_interests: List[str], job_interests: Dict) -> float:
        """Calculate interests compatibility"""
        if not user_interests:
            return 0.5
        
        total_score = 0
        max_possible = 0
        
        for interest, importance in job_interests.items():
            max_possible += importance
            if interest in user_interests:
                total_score += importance
        
        return (total_score / max_possible) if max_possible > 0 else 0

    def _calculate_work_styles_match(self, personality: Dict, job_work_styles: Dict) -> float:
        """Calculate work styles compatibility based on personality traits"""
        if not job_work_styles:
            return 0.5
        
        # Map personality traits to work styles
        personality_to_work_style = {
            "analytical_thinking": personality.get("openness", 3),
            "attention_to_detail": personality.get("conscientiousness", 3),
            "dependability": personality.get("conscientiousness", 3),
            "leadership": personality.get("extraversion", 3),
            "stress_tolerance": 5 - personality.get("neuroticism", 3),
            "adaptability": personality.get("openness", 3),
            "social_orientation": personality.get("extraversion", 3),
            "achievement": personality.get("conscientiousness", 3),
            "initiative": personality.get("extraversion", 3),
            "persistence": personality.get("conscientiousness", 3),
            "concern_for_others": personality.get("agreeableness", 3),
            "cooperation": personality.get("agreeableness", 3)
        }
        
        scores = []
        for style, importance in job_work_styles.items():
            user_level = personality_to_work_style.get(style, 3.0)
            
            norm_importance = importance / 5.0
            norm_user = user_level / 5.0
            
            if norm_user >= norm_importance * 0.8:
                score = 1.0
            else:
                gap = (norm_importance * 0.8) - norm_user
                score = max(0, 1 - (gap * 2))
            
            scores.append(score * norm_importance)
        
        return sum(scores) / sum(job_work_styles.values()) * 5 if scores else 0

    def _identify_strengths_improvements(self, profile: PersonProfile, job: Dict) -> Tuple[List[str], List[Dict]]:
        """Identify user strengths and areas for improvement"""
        strengths = []
        improvements = []
        
        skill_mapping = {
            "Programming": "programming",
            "Problem Solving": "problem_solving",
            "Public Speaking": "public_speaking",
            "Creative": "creative",
            "Working with People": "working_with_people",
            "Leadership": "leadership",
            "Networking": "networking",
            "Math": "math",
            "Tech-Savvy": "tech_savvy",
            "Empathy": "empathy",
            "Time Management": "time_management",
            "Attention to Detail": "attention_to_detail",
            "Project Management": "project_management",
            "Research": "research"
        }
        
        for skill_name in job["required_skills"]:
            skill_key = skill_mapping.get(skill_name, skill_name.lower().replace(' ', '_'))
            user_level = profile.skills.get(skill_key, 2.5)
            job_requirement = job["skills"].get(skill_key, 3.5)
            
            if user_level >= job_requirement:
                strengths.append(f"Strong {skill_name} skills (Level {user_level}/5)")
            elif user_level < job_requirement - 0.5:
                improvements.append({
                    "skill": skill_name,
                    "current_level": user_level,
                    "required_level": job_requirement,
                    "gap_severity": "High" if job_requirement - user_level > 1.5 else "Medium"
                })
        
        return strengths, improvements

    async def analyze_person_with_top_matches(self, profile: PersonProfile, top_n: int = 3) -> Dict:
        """
        MODIFIED METHOD: Analyze a person against only the top N job matches with AI insights
        This reduces API calls and focuses on most relevant careers
        """
        # Step 1: Get top N job matches (lightweight calculation)
        print(f"Calculating match scores for all {len(self.onet_jobs)} jobs...")
        top_job_names = self.get_top_job_matches(profile, top_n)
        print(f"Top {top_n} job matches identified: {', '.join(top_job_names)}")
        
        # Step 2: Generate detailed analysis with AI insights for top matches only
        matches = []
        for i, job_name in enumerate(top_job_names, 1):
            print(f"Generating AI insights for match {i}/{top_n}: {job_name}")
            
            # Calculate detailed match data
            match_result = self.calculate_job_match(profile, job_name)
            
            # Generate AI insights
            ai_insights = await self.generate_ai_insights(profile, job_name, match_result)
            
            # Combine results
            enhanced_match = {**match_result, **ai_insights}
            matches.append(enhanced_match)
        
        print(f"Analysis complete for top {top_n} matches.")
        
        return {
            "profile": asdict(profile),
            "matches": matches,
            "top_match": matches[0] if matches else None,
            "total_jobs_considered": len(self.onet_jobs),
            "jobs_analyzed_with_ai": len(matches),
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def generate_pdf_report(self, analysis_data: Dict, job_name: str) -> io.BytesIO:
        """Generate a comprehensive PDF report matching the Pookie style"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        avail_w = doc.width
        
        # Enhanced styles
        styles = getSampleStyleSheet()
        
        # Custom styles matching the sample
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=28,
            spaceAfter=10,
            textColor=colors.HexColor('#1a1a1a'),
            fontName='Helvetica-Bold'
        )
        
        email_style = ParagraphStyle(
            'EmailStyle',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=30,
            textColor=colors.HexColor('#666666')
        )

        cell_style = ParagraphStyle(
            'Cell',
            parent=styles['Normal'],
            fontSize=9,
            leading=12,             
            textColor=colors.black,
            spaceAfter=4,
            wordWrap='CJK'          
        )
        
        section_header_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading1'],
            fontSize=20,
            spaceAfter=15,
            textColor=colors.HexColor('#2E3440'),
            fontName='Helvetica-Bold'
        )
        
        subsection_style = ParagraphStyle(
            'SubsectionHeader',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=10,
            textColor=colors.HexColor('#5E81AC'),
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'BodyText',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            leading=14
        )
        
        # Find the specific job match
        job_match = None
        for match in analysis_data['matches']:
            if match['job_name'] == job_name:
                job_match = match
                break
        
        if not job_match:
            job_match = analysis_data['matches'][0]
        
        story = []
        
        # Header with name and email
        story.append(Paragraph(analysis_data['profile']['name'], title_style))
        story.append(Paragraph(analysis_data['profile']['email'], email_style))
        
        # Top 3 Careers Section
        story.append(Paragraph("Top 3 Career Matches", section_header_style))
        
        # Methodology explanation (similar to sample)
        methodology_text = """We have identified your top career matches using a sophisticated algorithm that integrates your preferences, 
        personality, and skills with five proven, industry-leading frameworks and assessments:<br/><br/>
        
        1. <b>RIASEC:</b> Your work-interest mix across six themes (Realistic, Investigative, Artistic, Social, Enterprising, Conventional)<br/>
        2. <b>OCEAN:</b> Your Big Five personality profile (Openness, Conscientiousness, Extraversion, Agreeableness, Emotional Stability)<br/>
        3. <b>Skills:</b> Your core strengths and learning modes (analytical, creative, technical)<br/>
        4. <b>Values:</b> What you want from work (Income, Impact, Stability, Variety, Recognition, Autonomy)<br/>
        5. <b>Direct Skills:</b> A 16-skill self-rating across everyday abilities matched directly to job requirements
        """
        story.append(Paragraph(methodology_text, body_style))
        story.append(Spacer(1, 20))
        
        # Top 3 careers table
        career_data = [[
            'Career', 'Overall %', 'Skills %', 'Values %', 'Interest %', 'Personality %', '1-line Why'
        ]]

        for i, match in enumerate(analysis_data['matches'][:3]):
            why_text = self._generate_one_line_why(match, analysis_data['profile'])
            career_data.append([
                Paragraph(match['job_name'], cell_style),
                f"{match['overall_match']}%",
                f"{match['breakdown']['skills_match']}%",
                f"{match['breakdown']['values_match']}%",
                f"{match['breakdown']['interests_match']}%",
                f"{match['breakdown']['work_styles_match']}%",
                Paragraph(why_text, cell_style)
            ])

        
        career_table = Table(
            career_data,
            colWidths=[
                0.26*avail_w,  # Career
                0.10*avail_w,  # Overall %
                0.10*avail_w,  # Skills %
                0.10*avail_w,  # Values %
                0.10*avail_w,  # Interest %
                0.10*avail_w,  # Personality %
                0.24*avail_w   # 1-line Why
            ]
        )
        career_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E8E8E8')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (1, 0), (5, -1), 'CENTER'),  # Center align percentages
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),    # Left align job names
            ('ALIGN', (6, 0), (6, -1), 'LEFT'),    # Left align why column
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        story.append(career_table)
        story.append(Spacer(1, 30))
        
        # Most Compatible Field Analysis (focused on selected job)
        story.append(Paragraph(f"Most Compatible Field: {job_match['job_name']}", subsection_style))
        story.append(Paragraph(f"Score: {job_match['overall_match']}%", body_style))
        story.append(Spacer(1, 10))
        
        # Strengths and Gaps in two columns
        strengths_gaps_data = [['Strengths', 'Gaps']]
        
        # Format strengths
        strengths_text = ""
        for s in job_match.get('strengths', [])[:4]:
            strengths_text += f"• {s}<br/>"

        gaps_text = ""
        improvements = job_match.get('improvements', [])[:3]
        if improvements:
            for imp in improvements:
                gaps_text += f"• {imp['skill']}: improve {imp['current_level']}/5 → {imp['required_level']}/5<br/>"
        else:
            gaps_text = ("• Strong alignment across key areas<br/>"
                        "• Minor refinements in specialized skills may boost advancement<br/>")

        strengths_gaps_data.append([
            Paragraph(strengths_text, cell_style),
            Paragraph(gaps_text, cell_style)
        ])

        
        strengths_gaps_table = Table(strengths_gaps_data, colWidths=[avail_w/2, avail_w/2])
        strengths_gaps_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F5F5F5')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10)
        ]))
        story.append(strengths_gaps_table)
        story.append(Spacer(1, 20))
        
        # Improvement Hacks
        story.append(Paragraph("Improvement Hacks:", subsection_style))
        action_plan = job_match.get('action_plan', {})
        if 'action_items' in action_plan:
            for item in action_plan['action_items'][:4]:
                story.append(Paragraph(f"• {item}", body_style))
        story.append(Spacer(1, 20))
        
        # Interview Tips
        story.append(Paragraph("Interview Tips", subsection_style))
        interview_insights = job_match.get('interview_insights', {})
        
        if 'key_selling_points' in interview_insights:
            story.append(Paragraph(f"• <b>Open with:</b> \"I'm a {self._create_opening_line(job_match, analysis_data['profile'])}\"", body_style))
            
            for i, point in enumerate(interview_insights['key_selling_points'][:3], 2):
                story.append(Paragraph(f"• <b>Point {i}:</b> {point}", body_style))
        
        if 'questions_to_ask' in interview_insights:
            story.append(Paragraph(f"• <b>Close with a fit test:</b> \"{interview_insights['questions_to_ask'][0]}\"", body_style))
        
        story.append(PageBreak())
        
        # Skills Analysis
        story.append(Paragraph("Skills", section_header_style))
        
        # What Works / What Doesn't table
        skills_analysis_data = [['What Works?', 'What Doesn\'t?']]
        
        # Find top skills and gaps
        user_skills = analysis_data['profile']['skills']
        top_skills = sorted(user_skills.items(), key=lambda x: x[1], reverse=True)[:3]
        weak_skills = sorted(user_skills.items(), key=lambda x: x[1])[:2]
        
        works_text = f"<b>Most-Matched Skill:</b> {top_skills[0][0].replace('_', ' ').title()} (Level {top_skills[0][1]}/5) — {self._get_skill_insight(top_skills[0][0], job_match)}<br/><br/>"
        works_text += f"<b>Secondary Strengths:</b> {', '.join([skill.replace('_', ' ').title() for skill, _ in top_skills[1:3]])}"
        
        doesnt_work_text = f"<b>Largest Gap Skill:</b> {weak_skills[0][0].replace('_', ' ').title()} (Level {weak_skills[0][1]}/5) — {self._get_improvement_insight(weak_skills[0][0])}<br/><br/>"
        doesnt_work_text += f"<b>Action:</b> Focus development on {weak_skills[0][0].replace('_', ' ').lower()} through targeted practice and learning."
        
        skills_works_p = Paragraph(works_text, cell_style)
        skills_doesnt_p = Paragraph(doesnt_work_text, cell_style)
        skills_analysis_data.append([skills_works_p, skills_doesnt_p])
        
        skills_table = Table(skills_analysis_data, colWidths=[avail_w/2, avail_w/2])
        skills_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F0F8FF')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15)
        ]))
        story.append(skills_table)
        story.append(Spacer(1, 25))
        
        # Top 5 Industries
        story.append(Paragraph("Top 5 Industries", subsection_style))
        industries = self._get_related_industries(job_match)
        for i, (industry, description) in enumerate(industries[:5], 1):
            story.append(Paragraph(f"{i}. <b>{industry}</b> — {description}", body_style))
        story.append(Spacer(1, 20))
        
        # Values Check
        story.append(Paragraph("Values Alignment Check", subsection_style))
        values_insight = self._generate_values_insight(job_match, analysis_data['profile'])
        story.append(Paragraph(values_insight, body_style))
        story.append(Spacer(1, 20))
        
        # Career Story
        story.append(Paragraph("Your Professional Narrative", subsection_style))
        career_story = job_match.get('career_story', 'Your career story showcases the unique combination of skills and experiences that make you an ideal candidate for this role.')
        story.append(Paragraph(career_story, body_style))
        
        # Footer
        story.append(Spacer(1, 30))
        story.append(Paragraph(f"Report generated on {analysis_data.get('analysis_date', datetime.now().strftime('%Y-%m-%d'))}", 
                            ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9, textColor=colors.gray)))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def _get_skill_insight(self, skill: str, job_match: Dict) -> str:
        """Generate insight about why a skill works well"""
        skill_insights = {
            'programming': 'Essential for technical roles and automation',
            'creative': 'Drives innovation and unique problem-solving approaches',
            'leadership': 'Critical for team management and project direction',
            'problem_solving': 'Core competency for analytical and strategic roles',
            'working_with_people': 'Vital for collaborative and client-facing positions',
            'math': 'Foundation for analytical and quantitative roles'
        }
        return skill_insights.get(skill, 'Valuable asset for professional success')

    def _generate_one_line_why(self, match: Dict, profile: Dict) -> str:
        """Generate a concise one-line explanation for job fit"""
        top_skills = sorted(profile['skills'].items(), key=lambda x: x[1], reverse=True)[:2]
        skills_text = f"{top_skills[0][0].replace('_', ' ')} + {top_skills[1][0].replace('_', ' ')}"
        
        personality = profile['personality']
        if personality.get('extraversion', 3) >= 4:
            personality_note = "leadership and visibility"
        elif personality.get('openness', 3) >= 4:
            personality_note = "innovation and creativity"
        else:
            personality_note = "analytical approach"
        
        return f"Perfect blend of {skills_text} skills with {personality_note}."
    
    def _get_improvement_insight(self, skill: str) -> str:
        """Generate insight about skill improvement"""
        improvement_insights = {
            'programming': 'Many modern roles expect basic coding literacy',
            'public_speaking': 'Essential for leadership and visibility',
            'networking': 'Critical for career advancement and opportunities',
            'tech_savvy': 'Increasingly important across all industries',
            'time_management': 'Fundamental for productivity and reliability'
        }
        return improvement_insights.get(skill, 'Important for well-rounded professional development')

    def _get_related_industries(self, job_match: Dict) -> List[Tuple[str, str]]:
        """Get related industries with descriptions"""
        job_name = job_match['job_name'].lower()
        
        industry_mappings = {
            'software': [
                ('Technology & Software', 'Direct fit with high growth potential and innovation'),
                ('Financial Services', 'FinTech and digital transformation opportunities'),
                ('Healthcare Technology', 'Growing sector with meaningful impact'),
                ('Consulting', 'Technical consulting and digital strategy'),
                ('Startups', 'High growth environment with diverse challenges')
            ],
            'marketing': [
                ('Advertising & Media', 'Creative campaigns and brand storytelling'),
                ('Technology', 'Product marketing and growth strategies'), 
                ('Consumer Goods', 'Brand management and market research'),
                ('Healthcare', 'Medical marketing and patient engagement'),
                ('Professional Services', 'B2B marketing and thought leadership')
            ],
            'analyst': [
                ('Financial Services', 'Investment analysis and risk management'),
                ('Consulting', 'Business analysis and strategic planning'),
                ('Technology', 'Data analysis and business intelligence'),
                ('Healthcare', 'Healthcare analytics and outcomes research'),
                ('Government', 'Policy analysis and public sector consulting')
            ]
        }
        
        # Default industries if no specific mapping found
        default_industries = [
            ('Professional Services', 'Consulting and advisory roles'),
            ('Technology', 'Innovation and digital transformation'),
            ('Financial Services', 'Analysis and strategic planning'),
            ('Healthcare', 'Meaningful impact and growth sector'),
            ('Education', 'Knowledge sharing and development')
        ]
        
        for key in industry_mappings:
            if key in job_name:
                return industry_mappings[key]
        
        return default_industries

    def _generate_values_insight(self, job_match: Dict, profile: Dict) -> str:
        """Generate insight about values alignment"""
        work_values = profile['work_values']
        top_value = max(work_values.items(), key=lambda x: x[1])
        
        value_job_fit = {
            'income': f"This role typically offers competitive compensation with growth potential.",
            'impact': f"Your work in {job_match['job_name']} will directly contribute to organizational success and meaningful outcomes.",
            'stability': f"{job_match['job_name']} roles offer strong job security and predictable career progression.",
            'variety': f"This position provides diverse challenges and project variety to keep you engaged.",
            'recognition': f"Success in {job_match['job_name']} roles is highly visible and valued by organizations.", 
            'autonomy': f"This role offers significant independence and decision-making authority."
        }
        
        insight = value_job_fit.get(top_value[0], f"Your top value ({top_value[0]}) aligns well with this career path.")
        return f"<b>{top_value[0].title()} Priority:</b> {insight}"

    def _create_opening_line(self, job_match: Dict, profile: Dict) -> str:
        """Create a compelling opening line for interviews"""
        job_name = job_match['job_name'].lower()
        top_skills = sorted(profile['skills'].items(), key=lambda x: x[1], reverse=True)[:2]
        
        skill_descriptors = {
            'programming': 'technical problem-solver',
            'creative': 'innovative thinker', 
            'leadership': 'results-driven leader',
            'problem_solving': 'analytical problem-solver',
            'working_with_people': 'collaborative professional',
            'math': 'quantitative analyst'
        }
        
        primary_descriptor = skill_descriptors.get(top_skills[0][0], 'dedicated professional')
        
        if 'analyst' in job_name:
            return f"{primary_descriptor} who turns complex data into actionable business insights"
        elif 'manager' in job_name or 'director' in job_name:
            return f"{primary_descriptor} who drives team success and delivers measurable results"
        elif 'developer' in job_name or 'engineer' in job_name:
            return f"{primary_descriptor} who builds scalable solutions and loves tackling technical challenges"
        else:
            return f"{primary_descriptor} passionate about creating value and driving meaningful outcomes"

# Initialize FastAPI app
app = FastAPI(title="AI-Enhanced Career Matching API - Top 3 Focus", version="2.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the AI matcher
ai_matcher = AICareerMatcher()

cred = credentials.Certificate({
    "type": os.getenv('FB_ACCOUNT_TYPE'),
    "project_id": os.getenv('FB_PROJECT_ID'),
    "private_key_id": os.getenv("FB_PRIVATE_KEY_ID"),
    "private_Key": os.getenv("FB_PRIVATE_KEY"),
    "client_email": os.getenv("FB_CLIENT_EMAIL"),
    "client_id": os.getenv("FB_CLIENT_ID"),
    "auth_uri":os.getenv("FB_AUTH_URI"),
    "token_uri": os.getenv("FB_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("FB_AUTH_CERT_URL"),
    "client_x509_cert_url": os.getenv("FB_CERT_URL"),
    "universe_domain": "googleapis.com"
})
firebase_admin.initialize_app(cred)
db = firestore.client()

@app.post("/analyze-profile-top3", response_model=AnalysisResponse)
async def analyze_profile_top_3_matches(request: PersonProfileRequest):
    """
    MODIFIED ENDPOINT: Analyze a person's profile and return AI insights for only the top 3 job matches
    This is more efficient and focused than analyzing all 15 jobs
    """

    data = request.model_dump(mode="json", exclude_none=True)
    data["created_at"] = firestore.SERVER_TIMESTAMP
    ref = db.collection("user").document()
    ref.set(data, merge=True)
    
    
    try:
        # Create profile from request
        profile = ai_matcher.create_profile_from_request(request)
        
        # Analyze only top 3 matches with AI insights (more efficient)
        result = await ai_matcher.analyze_person_with_top_matches(profile, top_n=3)
        
        return AnalysisResponse(
            success=True,
            message=f"Successfully analyzed top 3 career matches for {profile.name} with AI insights",
            result=result,
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing profile: {str(e)}")

@app.post("/analyze-profile-ai", response_model=AnalysisResponse)
async def analyze_profile_with_ai(request: PersonProfileRequest):
    """
    LEGACY ENDPOINT: Analyze all jobs (kept for backward compatibility)
    WARNING: This analyzes all 15 jobs and may be slower/more expensive
    """
    try:
        # Create profile from request
        profile = ai_matcher.create_profile_from_request(request)
        
        # Analyze all jobs (legacy method - more API calls)
        matches = []
        for job_name in ai_matcher.onet_jobs.keys():
            match_result = ai_matcher.calculate_job_match(profile, job_name)
            ai_insights = await ai_matcher.generate_ai_insights(profile, job_name, match_result)
            enhanced_match = {**match_result, **ai_insights}
            matches.append(enhanced_match)
        
        matches.sort(key=lambda x: x["overall_match"], reverse=True)
        
        result = {
            "profile": asdict(profile),
            "matches": matches,
            "top_match": matches[0] if matches else None,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return AnalysisResponse(
            success=True,
            message=f"Successfully analyzed profile for {profile.name} with AI insights (all {len(matches)} jobs)",
            result=result,
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing profile: {str(e)}")

@app.get("/quick-match-preview")
async def get_quick_match_preview(
    name: str,
    # Simplified parameters for quick preview
    math: int = 3,
    programming: int = 3,
    creative: int = 3,
    working_with_people: int = 3,
    leadership: int = 3
):
    """
    NEW ENDPOINT: Quick preview of top 3 matches without full analysis
    Useful for initial screening before full AI analysis
    """
    try:
        # Create minimal profile for quick matching
        minimal_skills = {
            "math": math,
            "programming": programming, 
            "creative": creative,
            "working_with_people": working_with_people,
            "leadership": leadership,
            "problem_solving": 3,
            "tech_savvy": 3,
            "teamwork": 3,
            "attention_to_detail": 3,
            "research": 3,
            "writing": 3,
            "public_speaking": 3,
            "networking": 3,
            "empathy": 3,
            "time_management": 3,
            "project_management": 3
        }
        
        minimal_profile = PersonProfile(
            name=name,
            email="preview@example.com",
            university="Preview",
            personality={"openness": 3, "conscientiousness": 3, "extraversion": 3, "agreeableness": 3, "neuroticism": 3},
            work_values={"income": 3, "impact": 3, "stability": 3, "variety": 3, "recognition": 3, "autonomy": 3},
            skills=minimal_skills,
            interests=["investigative"],
            preferred_career="Preview"
        )
        
        # Get top 3 matches quickly (no AI insights)
        top_jobs = ai_matcher.get_top_job_matches(minimal_profile, 3)
        quick_matches = []
        
        for job_name in top_jobs:
            match_data = ai_matcher.calculate_job_match(minimal_profile, job_name)
            quick_matches.append({
                "job_name": job_name,
                "overall_match": match_data["overall_match"],
                "skills_match": match_data["breakdown"]["skills_match"],
                "required_skills": match_data["required_skills"][:3]  # Top 3 skills
            })
        
        return {
            "success": True,
            "message": f"Quick match preview for {name}",
            "top_matches": quick_matches,
            "note": "This is a preview. Use /analyze-profile-top3 for full AI analysis."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quick preview: {str(e)}")

@app.get("/download-report/{job_name}")
async def download_career_report(job_name: str, analysis_data: str):
    """
    Download a PDF report for a specific job match
    Note: In production, you'd want to store analysis results and retrieve by ID
    """
    try:
        # Parse the analysis data (in production, retrieve from database by ID)
        import json
        import urllib.parse
        
        decoded_data = urllib.parse.unquote(analysis_data)
        analysis_dict = json.loads(decoded_data)
        
        # Generate PDF
        pdf_buffer = ai_matcher.generate_pdf_report(analysis_dict, job_name)
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(pdf_buffer.read()),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=career_report_{job_name.replace(' ', '_')}.pdf"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating PDF report: {str(e)}")

@app.post("/generate-job-insights")
async def generate_specific_job_insights(
    job_name: str,
    request: PersonProfileRequest
):
    """
    Generate AI insights for a specific job without full analysis
    """
    try:
        profile = ai_matcher.create_profile_from_request(request)
        match_data = ai_matcher.calculate_job_match(profile, job_name)
        ai_insights = await ai_matcher.generate_ai_insights(profile, job_name, match_data)
        
        return {
            "success": True,
            "job_name": job_name,
            "match_data": match_data,
            "ai_insights": ai_insights,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating job insights: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI-Enhanced Career Matching API - Top 3 Focus",
        "version": "2.1.0",
        "optimization": "Now focuses on top 3 job matches for efficient AI analysis",
        "features": [
            "Top 3 job match focus (efficient)",
            "AI-powered job fit summaries",
            "Personalized action plans", 
            "Interview preparation insights",
            "Career story generation",
            "PDF report downloads",
            "Quick match preview"
        ],
        "endpoints": {
            "/analyze-profile-top3": "POST - AI analysis of top 3 matches (RECOMMENDED)",
            "/analyze-profile-ai": "POST - Full AI analysis of all jobs (legacy, slower)",
            "/quick-match-preview": "GET - Quick preview without AI insights",
            "/generate-job-insights": "POST - Generate AI insights for specific job",
            "/download-report/{job_name}": "GET - Download PDF report",
            "/jobs": "GET - List available job types",
            "/health": "GET - Health check"
        },
        "efficiency_note": f"Database contains {len(ai_matcher.onet_jobs)} jobs. Top 3 analysis reduces API calls by ~80%."
    }

@app.get("/jobs")
async def get_jobs():
    """Get list of available job types with enhanced details"""
    jobs_info = {}
    for job_name, job_data in ai_matcher.onet_jobs.items():
        jobs_info[job_name] = {
            "onet_code": job_data["onet_code"],
            "required_skills": job_data["required_skills"],
            "similar_roles": job_data.get("similar_roles", []),
            "keywords": job_data.get("job_keywords", []),
            "top_work_values": sorted(job_data["work_values"].items(), key=lambda x: x[1], reverse=True)[:3],
            "top_interests": sorted(job_data["interests"].items(), key=lambda x: x[1], reverse=True)[:3]
        }
    
    return {
        "total_jobs": len(ai_matcher.onet_jobs),
        "available_jobs": list(ai_matcher.onet_jobs.keys()),
        "job_details": jobs_info,
        "ai_features": [
            "Personalized fit analysis",
            "Career development roadmap", 
            "Interview preparation guide",
            "Professional narrative development"
        ],
        "recommendation": "Use /analyze-profile-top3 for efficient analysis of best matches"
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with AI service status"""
    ai_status = "available"
    try:
        # Test OpenAI connection with a minimal request
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        ai_status = "connected"
    except Exception:
        ai_status = "unavailable"
    
    return {
        "status": "healthy",
        "ai_service": ai_status,
        "total_jobs_in_database": len(ai_matcher.onet_jobs),
        "optimization": "Top 3 matching active",
        "timestamp": datetime.now().isoformat(),
        "version": "2.1.0"
    }
