from typing import Dict, List, Tuple
from datetime import datetime
from dataclasses import asdict
import json
import careers as careers  # your existing careers module

from core.openai_client import openai_client
from models.schemas import PersonProfileRequest, PersonProfile

class AICareerMatcher:
    def __init__(self) -> None:
        self.onet_jobs = careers.onet_jobs

    # ---------- Construction ----------
    def create_profile_from_request(self, request: PersonProfileRequest) -> PersonProfile:
        personality = {
            "openness": request.openness,
            "conscientiousness": request.conscientiousness,
            "extraversion": request.extraversion,
            "agreeableness": request.agreeableness,
            "neuroticism": request.neuroticism,
        }
        work_values = {
            "income": 7 - request.income_importance,
            "impact": 7 - request.impact_importance,
            "stability": 7 - request.stability_importance,
            "variety": 7 - request.variety_importance,
            "recognition": 7 - request.recognition_importance,
            "autonomy": 7 - request.autonomy_importance,
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
            "negotiation": request.negotiation, 
            "creativity": request.creativity, 
            "programming": request.programming,
            "languages": request.languages,
            "empathy": request.empathy,
            "time_management": request.time_management,
            "attention_to_detail": request.attention_to_detail,
            "project_management": request.project_management,
            "artistic": request.artistic,
            "research": request.research,
            "hands_on_building": request.hands_on_building,
            "teamwork": request.teamwork,
        }
        return PersonProfile(
            name=request.name,
            email=request.email,
            university=request.university,
            personality=personality,
            work_values=work_values,
            skills=skills,
            interests=request.interests,
            preferred_career=request.preferred_career,
        )

    # ---------- Ranking ----------
    def get_top_job_matches(self, profile: PersonProfile, top_n: int = 3) -> List[str]:
        job_scores: List[Tuple[str, float]] = []
        for job_name, job in self.onet_jobs.items():
            skills_score = self._calculate_skills_match(profile.skills, job["skills"])
            values_score = self._calculate_values_match(profile.work_values, job["work_values"])
            interests_score = self._calculate_interests_match(profile.interests, job["interests"])
            work_styles_score = self._calculate_work_styles_match(profile.personality, job.get("work_styles", {}))
            overall = (skills_score * 0.3 + values_score * 0.25 + interests_score * 0.2 + work_styles_score * 0.25)
            job_scores.append((job_name, overall))
        job_scores.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in job_scores[:top_n]]

    async def analyze_person_with_top_matches(self, profile: PersonProfile, top_n: int = 3) -> Dict:
        top_names = self.get_top_job_matches(profile, top_n)
        matches: List[Dict] = []
        for job_name in top_names:
            match = self.calculate_job_match(profile, job_name)
            ai = await self.generate_ai_insights(profile, job_name, match)
            matches.append({**match, **ai})
        return {
            "profile": asdict(profile),
            "matches": matches,
            "top_match": matches[0] if matches else None,
            "total_jobs_considered": len(self.onet_jobs),
            "jobs_analyzed_with_ai": len(matches),
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    # ---------- AI ----------
    async def generate_ai_insights(self, profile: PersonProfile, job_name: str, match_data: Dict) -> Dict:
        job = self.onet_jobs[job_name]
        ai_summary = await self._generate_ai_summary(profile, job_name, job, match_data)
        keywords = await self._generate_keywords(job_name, job)
        onet_categories = await self._generate_onet_categories(job_name, job)
        action_plan = await self._generate_action_plan(profile, job_name, match_data)
        career_story = await self._generate_career_story(profile, job_name, match_data)
        interview_insights = await self._generate_interview_insights(profile, job_name, match_data)
        return {
            "ai_summary": ai_summary,
            "keywords": keywords,
            "onet_categories": onet_categories,
            "action_plan": action_plan,
            "career_story": career_story,
            "interview_insights": interview_insights,
            "similar_roles": job.get("similar_roles", []),
        }

    async def _generate_ai_summary(self, profile: PersonProfile, job_name: str, job: Dict, match: Dict) -> str:
        prompt = f"""
Create a personalized 2-3 paragraph summary for {profile.name} regarding their fit for the {job_name} position.
Overall {match['overall_match']}%. Skills {match['breakdown']['skills_match']}%, Values {match['breakdown']['values_match']}%, Interests {match['breakdown']['interests_match']}%, Work Styles {match['breakdown']['work_styles_match']}%.
Tone: encouraging, professional; acknowledge strengths, address gaps, motivate action.
"""
        try:
            r = openai_client.chat(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.7,
            )
            return r.choices[0].message.content.strip()
        except Exception:
            return (
                f"AI analysis temporarily unavailable. Based on your {match['overall_match']}% match, "
                "you show strong potential with several key strengths."
            )

    async def _generate_keywords(self, job_name: str, job: Dict) -> List[str]:
        if "job_keywords" in job:
            return job["job_keywords"][:4]
        try:
            r = openai_client.chat(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Return 4 concise keywords for {job_name}, comma-separated."}],
                max_tokens=50,
                temperature=0.5,
            )
            return [k.strip() for k in r.choices[0].message.content.strip().split(",")][:4]
        except Exception:
            return ["professional", "skilled", "dedicated", "growth-oriented"]

    async def _generate_onet_categories(self, job_name: str, job: Dict) -> Dict[str, List[str]]:
        categories = {
            "skills": list(job["skills"].keys())[:3],
            "work_values": list(job["work_values"].keys())[:3],
            "work_styles": list(job.get("work_styles", {}).keys())[:3],
            "interests": list(job["interests"].keys())[:3],
        }
        for k, items in categories.items():
            categories[k] = [i.replace("_", " ").title() for i in items]
        return categories

    async def _generate_action_plan(self, profile: PersonProfile, job_name: str, match: Dict) -> Dict:
        improvements = match.get("improvements", [])[:3]
        gap_lines = "\n".join(
            [f"- {imp['skill']}: gap {(imp['required_level'] - imp['current_level']):.1f}" for imp in improvements]
        )
        prompt = f"""
Create a focused action plan for {profile.name} to become competitive for {job_name}.
Current gaps:
{gap_lines}

Provide JSON with "top_needs" (3 items) and "action_items" (4-5 items).
"""
        try:
            r = openai_client.chat(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.6,
            )
            content = r.choices[0].message.content.strip()
            import re
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                return json.loads(m.group())
        except Exception:
            pass
        return {
            "top_needs": [
                f"Develop {improvements[0]['skill']} skills" if improvements else "Enhance core competencies",
                "Build relevant experience",
                "Network in target industry",
            ],
            "action_items": [
                "Take targeted online courses",
                "Ship a small portfolio project",
                "Informational interviews (4–6 people)",
                "Mock interviews focusing on gaps",
                "Document progress weekly",
            ],
        }

    async def _generate_career_story(self, profile: PersonProfile, job_name: str, match: Dict) -> str:
        prompt = f"""
Write a 2-paragraph first-person career story for {profile.name} targeting {job_name}.
Match score: {match['overall_match']}%. Be authentic, confident, forward-looking.
"""
        try:
            r = openai_client.chat(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7,
            )
            return r.choices[0].message.content.strip()
        except Exception:
            return (
                f"My journey toward {job_name} reflects strengths that align with the role. "
                "I'm excited to apply them and grow meaningful impact."
            )

    async def _generate_interview_insights(self, profile: PersonProfile, job_name: str, match: Dict) -> Dict:
        prompt = f"""
Create JSON interview prep for {profile.name} applying for {job_name} with:
- key_selling_points (3)
- story_examples (3)
- questions_interviewer_may_ask (3)
Be specific and actionable.
"""
        try:
            r = openai_client.chat(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
                temperature=0.6,
            )
            import re
            m = re.search(r"\{.*\}", r.choices[0].message.content.strip(), re.DOTALL)
            if m:
                return json.loads(m.group())
        except Exception:
            pass
        return {
            "key_selling_points": [
                "Strong analytical and problem-solving abilities",
                "Clear communication with stakeholders",
                "Fast learner with bias to action",
            ],
            "story_examples": [
                "Solved a complex data/tech problem under time constraints",
                "Led a cross-functional mini-project to delivery",
                "Learned a new tool quickly and applied it to real work",
            ],
            "questions_to_ask": [
                "How is success measured in the first 6–12 months?",
                "What current challenges is the team most focused on?",
                "How does this role contribute to the org's strategy?",
            ],
        }

    # ---------- Scoring ----------
    def calculate_job_match(self, profile: PersonProfile, job_name: str) -> Dict:
        job = self.onet_jobs[job_name]
        skills = self._calculate_skills_match(profile.skills, job["skills"])
        values = self._calculate_values_match(profile.work_values, job["work_values"])
        interests = self._calculate_interests_match(profile.interests, job["interests"])
        styles = self._calculate_work_styles_match(profile.personality, job.get("work_styles", {}))
        overall = (skills * 0.3 + values * 0.25 + interests * 0.2 + styles * 0.25) * 100
        strengths, improvements = self._identify_strengths_improvements(profile, job)
        return {
            "job_name": job_name,
            "onet_code": job["onet_code"],
            "overall_match": round(overall, 1),
            "breakdown": {
                "skills_match": round(skills * 100, 1),
                "values_match": round(values * 100, 1),
                "interests_match": round(interests * 100, 1),
                "work_styles_match": round(styles * 100, 1),
            },
            "strengths": strengths,
            "improvements": improvements,
            "required_skills": job["required_skills"],
        }

    def _calculate_skills_match(self, user_skills: Dict, job_skills: Dict) -> float:
        scores, weights = [], []
        for skill, importance in job_skills.items():
            u = user_skills.get(skill, 2.5) / 5.0
            w = importance / 5.0
            score = 1.0 if u >= w else max(0, 1 - ((w - u) * 2))
            scores.append(score * w)
            weights.append(importance)
        return (sum(scores) / sum(weights) * 5) if scores else 0

    def _calculate_values_match(self, user_values: Dict, job_values: Dict) -> float:
        scores, weights = [], []
        for k, j_imp in job_values.items():
            u_imp = user_values.get(k, 3.5)
            nj, nu = j_imp / 5.0, u_imp / 6.0
            sim = 1 - abs(nj - nu)
            scores.append(sim * nj)
            weights.append(j_imp)
        return (sum(scores) / sum(weights) * 5) if scores else 0

    def _calculate_interests_match(self, user_interests: List[str], job_interests: Dict) -> float:
        if not user_interests: return 0.5
        max_possible = sum(job_interests.values())
        got = sum(v for k, v in job_interests.items() if k in user_interests)
        return (got / max_possible) if max_possible else 0

    def _calculate_work_styles_match(self, personality: Dict, job_styles: Dict) -> float:
        if not job_styles: return 0.5
        map_ps = {
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
            "cooperation": personality.get("agreeableness", 3),
        }
        scores, weights = [], []
        for style, imp in job_styles.items():
            u = map_ps.get(style, 3) / 5.0
            w = imp / 5.0
            thresh = w * 0.8
            score = 1.0 if u >= thresh else max(0, 1 - ((thresh - u) * 2))
            scores.append(score * w)
            weights.append(imp)
        return (sum(scores) / sum(weights) * 5) if scores else 0

    def _identify_strengths_improvements(self, profile: PersonProfile, job: Dict) -> Tuple[List[str], List[Dict]]:
        strengths, improvements = [], []
        # Updated mapping to include the 5 new skills
        mapping = {
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
            "Research": "research",
            # New skill mappings
            "Negotiation": "negotiation",
            "Creativity": "creativity",
            "Languages": "languages",
            "Artistic": "artistic",
            "Hands-On Building": "hands_on_building",
            "Hands On Building": "hands_on_building",  # Alternative spacing
            # Additional common variations that might appear in job data
            "Language Skills": "languages",
            "Foreign Languages": "languages",
            "Multilingual": "languages",
            "Art": "artistic",
            "Design": "artistic",
            "Visual Arts": "artistic",
            "Manual Skills": "hands_on_building",
            "Construction": "hands_on_building",
            "Building": "hands_on_building",
            "Craftsmanship": "hands_on_building",
            "Creative Thinking": "creativity",
            "Innovation": "creativity",
            "Creative Problem Solving": "creativity",
        }
        
        for skill_name in job["required_skills"]:
            # Try direct mapping first
            key = mapping.get(skill_name)
            
            # If no direct mapping, try lowercase with underscores
            if key is None:
                key = skill_name.lower().replace(" ", "_").replace("-", "_")
            
            user = profile.skills.get(key, 2.5)
            need = job["skills"].get(key, 3.5)
            
            if user >= need:
                strengths.append(f"Strong {skill_name} skills (Level {user}/5)")
            elif user < need - 0.5:
                improvements.append({
                    "skill": skill_name,
                    "current_level": user,
                    "required_level": need,
                    "gap_severity": "High" if (need - user) > 1.5 else "Medium",
                })
                
        return strengths, improvements