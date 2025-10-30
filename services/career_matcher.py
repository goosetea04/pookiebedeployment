from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import asdict
import json
import math
import numpy as np
import logging
from collections import defaultdict
import careers as careers  # Existing careers module

logger =logging.getLogger(__name__)

from core.openai_client import openai_client
from models.schemas import PersonProfileRequest, PersonProfile

class AICareerMatcher:
    def __init__(self) -> None:
        self.onet_jobs = careers.onet_jobs
        self._build_skill_index()
        self._build_career_clusters()
        self._build_competence_learning_mappings()  # New

    def _build_skill_index(self):
        """Build reverse index of skills to jobs for better matching"""
        self.skill_to_jobs = defaultdict(list)
        self.all_skills = set()
        
        for job_name, job_data in self.onet_jobs.items():
            for skill in job_data.get("skills", {}):
                self.skill_to_jobs[skill].append(job_name)
                self.all_skills.add(skill)

    def _build_career_clusters(self):
        """Group careers by O*NET code prefixes for better recommendations"""
        self.career_clusters = defaultdict(list)
        
        for job_name, job_data in self.onet_jobs.items():
            onet_code = job_data.get("onet_code", "")
            if onet_code:
                # Group by first 2 digits (major occupation group)
                cluster = onet_code[:2]
                self.career_clusters[cluster].append(job_name)
    
    def _build_competence_learning_mappings(self):
        """Build mappings for dominant competence and learning styles"""
        # Map competences to skill categories
        self.competence_skill_map = {
            "Analytical": ["math", "problem_solving", "research", "attention_to_detail"],
            "Practical": ["hands_on_building", "tech_savvy", "attention_to_detail"],
            "Creative": ["creative", "innovation", "artistic", "writing"],
            "People": ["working_with_people", "empathy", "public_speaking", "leadership", "teamwork"]
        }
        
        # Map learning styles to work preferences
        self.learning_style_preferences = {
            "Research/Reading": {
                "preferred_skills": ["research", "writing", "problem_solving"],
                "work_context": "analytical and knowledge-intensive"
            },
            "Hands-on/Systems": {
                "preferred_skills": ["hands_on_building", "tech_savvy", "programming"],
                "work_context": "technical and practical"
            },
            "Teamwork/Interviewing": {
                "preferred_skills": ["working_with_people", "teamwork", "networking", "empathy"],
                "work_context": "collaborative and interpersonal"
            },
            "Brainstorming/Ideation": {
                "preferred_skills": ["creative", "innovation", "problem_solving"],
                "work_context": "innovative and strategic"
            }
        }

    # ---------- Enhanced Profile Creation ----------
    def create_profile_from_request(self, request: PersonProfileRequest) -> PersonProfile:
        """Enhanced profile creation with better skill normalization"""
        personality = {
            "openness": request.openness,
            "conscientiousness": request.conscientiousness,
            "extraversion": request.extraversion,
            "agreeableness": request.agreeableness,
            "neuroticism": request.neuroticism,
        }
        
        # Inverted work values (7 - x) to match your current system
        work_values = {
            "income": 7 - request.income_importance,
            "impact": 7 - request.impact_importance,
            "stability": 7 - request.stability_importance,
            "variety": 7 - request.variety_importance,
            "recognition": 7 - request.recognition_importance,
            "autonomy": 7 - request.autonomy_importance,
        }
        
        # Comprehensive skills mapping
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
            "innovation": request.innovation,
            "innovation": request.innovation,
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
            dominant_competence=request.dominant_competence,  # NEW
            learning_style=request.learning_style,  # NEW
        )

    # ---------- Enhanced Ranking with Multiple Algorithms ----------
    def get_top_job_matches(self, profile: PersonProfile, top_n: int = 10, algorithm: str = "comprehensive") -> List[Tuple[str, float, Dict]]:
        """Multiple matching algorithms for different use cases"""
        
        if algorithm == "skills_focused":
            return self._get_skills_focused_matches(profile, top_n)
        elif algorithm == "values_focused":
            return self._get_values_focused_matches(profile, top_n)
        elif algorithm == "hybrid":
            return self._get_hybrid_matches(profile, top_n)
        else:  # comprehensive (default)
            return self._get_comprehensive_matches(profile, top_n)

    def _get_comprehensive_matches(self, profile: PersonProfile, top_n: int) -> List[Tuple[str, float, Dict]]:
        """Enhanced comprehensive matching with detailed scoring including competence and learning style"""
        job_scores: List[Tuple[str, float, Dict]] = []
        
        for job_name, job_data in self.onet_jobs.items():
            # Calculate individual scores
            skills_result = self._calculate_enhanced_skills_match(profile.skills, job_data.get("skills", {}))
            values_result = self._calculate_enhanced_values_match(profile.work_values, job_data.get("work_values", {}))
            interests_result = self._calculate_enhanced_interests_match(profile.interests, job_data.get("interests", {}))
            work_styles_result = self._calculate_enhanced_work_styles_match(profile.personality, job_data.get("work_styles", {}))
            
            # NEW: Calculate competence and learning style matches
            competence_result = self._calculate_competence_match(
                profile.dominant_competence, 
                job_data.get("dominant_competence", []),
                job_data.get("skills", {})
            )
            learning_style_result = self._calculate_learning_style_match(
                profile.learning_style,
                job_data.get("learning_style", []),
                job_data
            )
            
            # Updated weights to include new factors
            base_weights = {
                "skills": 0.25,
                "values": 0.20,
                "interests": 0.15,
                "work_styles": 0.18,
                "competence": 0.12,  # NEW
                "learning_style": 0.10  # NEW
            }
            
            # Adaptive weighting
            weights = self._calculate_adaptive_weights_v2(
                job_data, skills_result, values_result, interests_result, 
                work_styles_result, competence_result, learning_style_result, base_weights
            )
            
            # Calculate weighted overall score
            overall_score = (
                skills_result["score"] * weights["skills"] +
                values_result["score"] * weights["values"] +
                interests_result["score"] * weights["interests"] +
                work_styles_result["score"] * weights["work_styles"] +
                competence_result["score"] * weights["competence"] +
                learning_style_result["score"] * weights["learning_style"]
            )
            
            # Bonus for career preference alignment
            preference_bonus = self._calculate_preference_bonus(profile.preferred_career, job_name, job_data)
            overall_score += preference_bonus
            
            # Penalty for severe skill gaps
            gap_penalty = self._calculate_gap_penalty(profile.skills, job_data.get("skills", {}))
            overall_score -= gap_penalty
            
            scoring_details = {
                "skills": skills_result,
                "values": values_result,
                "interests": interests_result,
                "work_styles": work_styles_result,
                "competence": competence_result,  # NEW
                "learning_style": learning_style_result,  # NEW
                "weights": weights,
                "preference_bonus": preference_bonus,
                "gap_penalty": gap_penalty,
                "raw_score": overall_score
            }
            
            job_scores.append((job_name, overall_score, scoring_details))
        
        # Sort by score and return top matches
        job_scores.sort(key=lambda x: x[1], reverse=True)
        return job_scores[:top_n]

    def _calculate_adaptive_weights_v2(self, job_data: Dict, skills_result: Dict, values_result: Dict, 
                                     interests_result: Dict, work_styles_result: Dict, competence_result: Dict, 
                                     learning_style_result: Dict, base_weights: Dict) -> Dict[str, float]:
        """Dynamically adjust weights based on data quality and availability - V2 with competence/learning"""
        
        # Adjust based on data completeness
        skills_completeness = len(job_data.get("skills", {})) / 6.0
        values_completeness = len(job_data.get("work_values", {})) / 6.0
        interests_completeness = len(job_data.get("interests", {})) / 6.0
        work_styles_completeness = len(job_data.get("work_styles", {})) / 6.0
        competence_completeness = 1.0 if job_data.get("dominant_competence") else 0.5
        learning_style_completeness = 1.0 if job_data.get("learning_style") else 0.5
        
        # Adjust based on confidence scores
        confidence_multipliers = {
            "skills": skills_result.get("confidence", 1.0) * skills_completeness,
            "values": values_result.get("confidence", 1.0) * values_completeness,
            "interests": interests_result.get("confidence", 1.0) * interests_completeness,
            "work_styles": work_styles_result.get("confidence", 1.0) * work_styles_completeness,
            "competence": competence_result.get("confidence", 1.0) * competence_completeness,
            "learning_style": learning_style_result.get("confidence", 1.0) * learning_style_completeness,
        }
        
        # Normalize weights
        total_weight = sum(base_weights[k] * confidence_multipliers[k] for k in base_weights)
        
        return {
            k: (base_weights[k] * confidence_multipliers[k]) / total_weight 
            for k in base_weights
        }

    def _calculate_competence_match(self, user_competences: List[str], job_competences: List[str], job_skills: Dict) -> Dict:
        """Calculate match based on dominant competences"""
        if not user_competences:
            return {"score": 0.5, "confidence": 0.2, "matches": [], "mismatches": []}
        
        # Direct competence matching
        direct_matches = []
        mismatches = []
        
        for user_comp in user_competences:
            if user_comp in job_competences:
                direct_matches.append({
                    "competence": user_comp,
                    "match_type": "direct",
                    "strength": 1.0
                })
            else:
                mismatches.append({
                    "user_competence": user_comp,
                    "job_has": job_competences
                })
        
        # Skill-based competence validation
        skill_alignment_score = 0
        for user_comp in user_competences:
            related_skills = self.competence_skill_map.get(user_comp, [])
            
            # Check how many related skills are required by the job
            relevant_job_skills = {skill: level for skill, level in job_skills.items() if skill in related_skills}
            
            if relevant_job_skills:
                avg_importance = sum(relevant_job_skills.values()) / len(relevant_job_skills)
                skill_alignment_score += avg_importance / 5.0  # Normalize to 0-1
        
        # Calculate final score
        direct_match_score = len(direct_matches) / len(user_competences) if user_competences else 0
        skill_validation_score = skill_alignment_score / len(user_competences) if user_competences else 0
        
        # Weighted combination: 60% direct match, 40% skill validation
        final_score = (direct_match_score * 0.6) + (skill_validation_score * 0.4)
        
        confidence = 0.9 if job_competences else 0.5  # High confidence if job has competence data
        
        return {
            "score": min(final_score, 1.0),
            "confidence": confidence,
            "matches": direct_matches,
            "mismatches": mismatches,
            "direct_match_rate": direct_match_score,
            "skill_alignment": skill_validation_score
        }

    def _calculate_learning_style_match(self, user_learning_styles: List[str], job_learning_styles: List[str], job_data: Dict) -> Dict:
        """Calculate job fit based on learning style preferences"""
        if not user_learning_styles:
            return {"score": 0.5, "confidence": 0.2, "matches": [], "context_fit": []}
        
        # Direct learning style matching
        direct_matches = []
        context_fits = []
        
        for user_style in user_learning_styles:
            if user_style in job_learning_styles:
                direct_matches.append({
                    "learning_style": user_style,
                    "match_type": "direct",
                    "strength": 1.0
                })
            
            # Check skill alignment with learning style preferences
            style_prefs = self.learning_style_preferences.get(user_style, {})
            preferred_skills = style_prefs.get("preferred_skills", [])
            work_context = style_prefs.get("work_context", "")
            
            # Calculate how well job skills align with this learning style
            job_skills = job_data.get("skills", {})
            alignment_score = 0
            
            for skill in preferred_skills:
                if skill in job_skills:
                    alignment_score += job_skills[skill] / 5.0
            
            if preferred_skills:
                avg_alignment = alignment_score / len(preferred_skills)
                context_fits.append({
                    "learning_style": user_style,
                    "work_context": work_context,
                    "skill_alignment": avg_alignment,
                    "fit_level": "High" if avg_alignment > 0.7 else "Medium" if avg_alignment > 0.4 else "Low"
                })
        
        # Calculate final score
        direct_match_score = len(direct_matches) / len(user_learning_styles) if user_learning_styles else 0
        
        # Average context fit score
        context_fit_score = 0
        if context_fits:
            context_fit_score = sum(cf["skill_alignment"] for cf in context_fits) / len(context_fits)
        
        # Weighted combination: 50% direct match, 50% context fit
        final_score = (direct_match_score * 0.5) + (context_fit_score * 0.5)
        
        confidence = 0.8 if job_learning_styles else 0.5
        
        return {
            "score": min(final_score, 1.0),
            "confidence": confidence,
            "matches": direct_matches,
            "context_fit": context_fits,
            "direct_match_rate": direct_match_score,
            "context_fit_score": context_fit_score
        }

    def _calculate_enhanced_skills_match(self, user_skills: Dict[str, float], job_skills: Dict[str, float]) -> Dict:
        """Enhanced skills matching with gap analysis and confidence scoring"""
        if not job_skills:
            return {"score": 0.5, "confidence": 0.1, "gaps": [], "strengths": []}
        
        matches = []
        gaps = []
        strengths = []
        total_weight = 0
        weighted_score = 0
        
        for skill, required_level in job_skills.items():
            user_level = user_skills.get(skill, 2.5)  # Default to middle if not specified
            weight = required_level / 5.0  # Normalize to 0-1
            
            # Calculate skill-specific match
            if user_level >= required_level:
                skill_score = 1.0
                strengths.append({
                    "skill": skill,
                    "user_level": user_level,
                    "required_level": required_level,
                    "surplus": user_level - required_level
                })
            else:
                gap = required_level - user_level
                # Exponential decay for gaps (severe gaps hurt more)
                skill_score = max(0, math.exp(-gap * 0.5))
                gaps.append({
                    "skill": skill,
                    "user_level": user_level,
                    "required_level": required_level,
                    "gap": gap,
                    "severity": "Critical" if gap > 2.0 else "High" if gap > 1.0 else "Medium"
                })
            
            matches.append(skill_score)
            weighted_score += skill_score * weight
            total_weight += weight
        
        final_score = weighted_score / total_weight if total_weight > 0 else 0
        confidence = len(job_skills) / 6.0  # Higher confidence with more skills data
        
        return {
            "score": final_score,
            "confidence": min(confidence, 1.0),
            "gaps": sorted(gaps, key=lambda x: x["gap"], reverse=True),
            "strengths": sorted(strengths, key=lambda x: x["surplus"], reverse=True),
            "skill_coverage": len([s for s in job_skills if s in user_skills]) / len(job_skills)
        }

    def _calculate_enhanced_values_match(self, user_values: Dict[str, float], job_values: Dict[str, float]) -> Dict:
        """Enhanced values matching with importance weighting"""
        if not job_values:
            return {"score": 0.5, "confidence": 0.1, "alignments": [], "conflicts": []}
        
        alignments = []
        conflicts = []
        total_weight = 0
        weighted_score = 0
        
        for value, job_importance in job_values.items():
            user_importance = user_values.get(value, 3.5)
            weight = job_importance / 5.0
            
            # Normalize both to 0-1 scale
            job_norm = job_importance / 5.0
            user_norm = user_importance / 6.0  # User values are 1-6 scale
            
            # Calculate similarity (1 - absolute difference)
            similarity = 1 - abs(job_norm - user_norm)
            
            if similarity >= 0.7:
                alignments.append({
                    "value": value,
                    "user_importance": user_importance,
                    "job_importance": job_importance,
                    "alignment": similarity
                })
            elif similarity < 0.4:
                conflicts.append({
                    "value": value,
                    "user_importance": user_importance,
                    "job_importance": job_importance,
                    "conflict_level": "High" if similarity < 0.2 else "Medium"
                })
            
            weighted_score += similarity * weight
            total_weight += weight
        
        final_score = weighted_score / total_weight if total_weight > 0 else 0
        confidence = len(job_values) / 6.0
        
        return {
            "score": final_score,
            "confidence": min(confidence, 1.0),
            "alignments": sorted(alignments, key=lambda x: x["alignment"], reverse=True),
            "conflicts": sorted(conflicts, key=lambda x: x.get("conflict_level", "Low"), reverse=True)
        }

    def _calculate_enhanced_interests_match(self, user_interests: List[str], job_interests: Dict[str, float]) -> Dict:
        """Enhanced RIASEC interests matching"""
        if not job_interests or not user_interests:
            return {"score": 0.3, "confidence": 0.2, "matches": [], "top_job_interests": []}
        
        # Normalize job interests
        total_job_interest = sum(job_interests.values())
        if total_job_interest == 0:
            return {"score": 0.3, "confidence": 0.1, "matches": [], "top_job_interests": []}
        
        matched_score = 0
        matches = []
        
        for interest in user_interests:
            if interest in job_interests:
                score = job_interests[interest] / 5.0  # Normalize to 0-1
                matched_score += score
                matches.append({
                    "interest": interest,
                    "job_importance": job_interests[interest],
                    "weight": score
                })
        
        # Get top job interests for comparison
        top_job_interests = sorted(
            job_interests.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        # Final score: ratio of matched interests weighted by importance
        max_possible = sum(sorted(job_interests.values(), reverse=True)[:len(user_interests)])
        final_score = matched_score / (max_possible / 5.0) if max_possible > 0 else 0
        
        confidence = min(len(job_interests) / 6.0, 1.0)
        
        return {
            "score": min(final_score, 1.0),
            "confidence": confidence,
            "matches": sorted(matches, key=lambda x: x["weight"], reverse=True),
            "top_job_interests": [(interest, score) for interest, score in top_job_interests],
            "coverage": len(matches) / len(user_interests) if user_interests else 0
        }

    def _calculate_enhanced_work_styles_match(self, personality: Dict[str, float], job_styles: Dict[str, float]) -> Dict:
        """Enhanced work styles matching with personality mapping"""
        if not job_styles:
            return {"score": 0.5, "confidence": 0.1, "matches": [], "gaps": []}
        
        # Enhanced personality to work style mapping
        personality_mapping = {
            "analytical_thinking": ["openness", "conscientiousness"],
            "attention_to_detail": ["conscientiousness"],
            "dependability": ["conscientiousness", "agreeableness"],
            "leadership": ["extraversion", "openness"],
            "stress_tolerance": ["neuroticism"],  # Inverted
            "adaptability": ["openness", "extraversion"],
            "social_orientation": ["extraversion", "agreeableness"],
            "achievement": ["conscientiousness", "extraversion"],
            "initiative": ["extraversion", "openness"],
            "persistence": ["conscientiousness"],
            "concern_for_others": ["agreeableness"],
            "cooperation": ["agreeableness", "extraversion"],
            "innovation": ["openness"],
            "independence": ["openness", "extraversion"]
        }
        
        matches = []
        gaps = []
        total_weight = 0
        weighted_score = 0
        
        for style, required_level in job_styles.items():
            # Calculate user level for this work style
            if style in personality_mapping:
                personality_traits = personality_mapping[style]
                if style == "stress_tolerance":
                    # Stress tolerance is inverse of neuroticism
                    user_level = 5 - personality.get("neuroticism", 3)
                else:
                    # Average the relevant personality traits
                    user_level = sum(personality.get(trait, 3) for trait in personality_traits) / len(personality_traits)
            else:
                user_level = 3.0  # Default neutral
            
            weight = required_level / 5.0
            
            # Calculate match score
            if user_level >= required_level * 0.8:  # 80% threshold
                score = min(1.0, user_level / required_level)
                matches.append({
                    "style": style,
                    "user_level": user_level,
                    "required_level": required_level,
                    "match_score": score
                })
            else:
                gap = required_level - user_level
                score = max(0, 1 - (gap / required_level))
                gaps.append({
                    "style": style,
                    "user_level": user_level,
                    "required_level": required_level,
                    "gap": gap
                })
            
            weighted_score += score * weight
            total_weight += weight
        
        final_score = weighted_score / total_weight if total_weight > 0 else 0
        confidence = len(job_styles) / 6.0
        
        return {
            "score": final_score,
            "confidence": min(confidence, 1.0),
            "matches": sorted(matches, key=lambda x: x["match_score"], reverse=True),
            "gaps": sorted(gaps, key=lambda x: x["gap"], reverse=True)
        }

    def _calculate_preference_bonus(self, preferred_career: Optional[str], job_name: str, job_data: Dict) -> float:
        """Bonus for jobs matching user's preferred career"""
        if not preferred_career:
            return 0.0
        
        preferred_lower = preferred_career.lower()
        job_lower = job_name.lower()
        
        # Direct match
        if preferred_lower in job_lower or job_lower in preferred_lower:
            return 0.1
        
        # Check similar roles
        similar_roles = job_data.get("similar_roles", [])
        for role in similar_roles:
            if preferred_lower in role.lower() or role.lower() in preferred_lower:
                return 0.05
        
        return 0.0

    def _calculate_gap_penalty(self, user_skills: Dict[str, float], job_skills: Dict[str, float]) -> float:
        """Penalty for severe skill gaps in critical areas"""
        if not job_skills:
            return 0.0
        
        penalty = 0.0
        for skill, required in job_skills.items():
            if required >= 4.0:  # Critical skill
                user_level = user_skills.get(skill, 2.5)
                if user_level < required - 1.5:  # Severe gap
                    penalty += 0.02
        
        return min(penalty, 0.1)  # Cap penalty at 10%

    # ---------- Alternative Matching Algorithms ----------
    def _get_skills_focused_matches(self, profile: PersonProfile, top_n: int) -> List[Tuple[str, float, Dict]]:
        """Skills-first matching for technical roles"""
        job_scores = []
        
        for job_name, job_data in self.onet_jobs.items():
            skills_result = self._calculate_enhanced_skills_match(profile.skills, job_data.get("skills", {}))
            values_result = self._calculate_enhanced_values_match(profile.work_values, job_data.get("work_values", {}))
            
            # Heavy weight on skills, light on values
            overall_score = skills_result["score"] * 0.8 + values_result["score"] * 0.2
            
            scoring_details = {
                "algorithm": "skills_focused",
                "skills": skills_result,
                "values": values_result,
                "raw_score": overall_score
            }
            
            job_scores.append((job_name, overall_score, scoring_details))
        
        job_scores.sort(key=lambda x: x[1], reverse=True)
        return job_scores[:top_n]

    def _get_values_focused_matches(self, profile: PersonProfile, top_n: int) -> List[Tuple[str, float, Dict]]:
        """Values-first matching for culture fit"""
        job_scores = []
        
        for job_name, job_data in self.onet_jobs.items():
            values_result = self._calculate_enhanced_values_match(profile.work_values, job_data.get("work_values", {}))
            interests_result = self._calculate_enhanced_interests_match(profile.interests, job_data.get("interests", {}))
            work_styles_result = self._calculate_enhanced_work_styles_match(profile.personality, job_data.get("work_styles", {}))
            
            # Heavy weight on values and culture fit
            overall_score = (values_result["score"] * 0.5 + 
                           interests_result["score"] * 0.3 + 
                           work_styles_result["score"] * 0.2)
            
            scoring_details = {
                "algorithm": "values_focused",
                "values": values_result,
                "interests": interests_result,
                "work_styles": work_styles_result,
                "raw_score": overall_score
            }
            
            job_scores.append((job_name, overall_score, scoring_details))
        
        job_scores.sort(key=lambda x: x[1], reverse=True)
        return job_scores[:top_n]

    def _get_hybrid_matches(self, profile: PersonProfile, top_n: int) -> List[Tuple[str, float, Dict]]:
        """Hybrid approach combining multiple algorithms"""
        comprehensive_matches = self._get_comprehensive_matches(profile, top_n * 2)
        skills_matches = self._get_skills_focused_matches(profile, top_n * 2)
        values_matches = self._get_values_focused_matches(profile, top_n * 2)
        
        # Combine scores using weighted average
        job_combined_scores = defaultdict(list)
        
        for matches, weight in [(comprehensive_matches, 0.5), (skills_matches, 0.3), (values_matches, 0.2)]:
            for job_name, score, details in matches:
                job_combined_scores[job_name].append((score, weight, details))
        
        # Calculate final hybrid scores
        final_scores = []
        for job_name, score_data in job_combined_scores.items():
            if len(score_data) >= 2:  # Must appear in at least 2 algorithms
                weighted_score = sum(score * weight for score, weight, _ in score_data)
                combined_details = {
                    "algorithm": "hybrid",
                    "component_scores": score_data,
                    "raw_score": weighted_score
                }
                final_scores.append((job_name, weighted_score, combined_details))
        
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[:top_n]

    # ---------- Enhanced Analysis ----------
    async def analyze_person_with_top_matches(self, profile: PersonProfile, top_n: int = 5, algorithm: str = "comprehensive") -> Dict:
        """Enhanced analysis with multiple algorithms and detailed insights"""
        top_matches = self.get_top_job_matches(profile, top_n, algorithm)
        
        enhanced_matches = []
        for job_name, score, scoring_details in top_matches:
            # Calculate traditional match for backward compatibility
            traditional_match = self.calculate_job_match(profile, job_name)
            
            # Generate AI insights
            ai_insights = await self.generate_enhanced_ai_insights(profile, job_name, traditional_match, scoring_details)
            
            enhanced_match = {
                **traditional_match,
                "enhanced_score": round(score * 100, 1),
                "algorithm_used": algorithm,
                "scoring_details": scoring_details,
                **ai_insights
            }
            
            enhanced_matches.append(enhanced_match)
        
        # Generate comparative analysis
        comparative_analysis = await self._generate_comparative_analysis(profile, enhanced_matches)
        
        return {
            "profile": asdict(profile),
            "matches": enhanced_matches,
            "top_match": enhanced_matches[0] if enhanced_matches else None,
            "comparative_analysis": comparative_analysis,
            "algorithm_used": algorithm,
            "total_jobs_considered": len(self.onet_jobs),
            "jobs_analyzed_with_ai": len(enhanced_matches),
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "recommendations": await self._generate_strategic_recommendations(profile, enhanced_matches)
        }

    async def _generate_comparative_analysis(self, profile: PersonProfile, matches: List[Dict]) -> Dict:
        """Generate comparative analysis across top matches"""
        if len(matches) < 2:
            return {}
        
        # Analyze skill patterns
        skill_analysis = self._analyze_skill_patterns(matches)
        
        # Analyze career progression paths
        progression_paths = self._analyze_progression_paths(matches)
        
        # Generate AI summary of choices
        comparison_summary = await self._generate_ai_comparison(profile, matches)
        
        return {
            "skill_patterns": skill_analysis,
            "progression_paths": progression_paths,
            "decision_framework": comparison_summary,
            "diversity_score": self._calculate_match_diversity(matches)
        }

    def _analyze_skill_patterns(self, matches: List[Dict]) -> Dict:
        """Analyze common skill patterns across matches"""
        all_skills = set()
        skill_frequencies = defaultdict(int)
        
        for match in matches:
            job_name = match["job_name"]
            job_data = self.onet_jobs.get(job_name, {})
            skills = job_data.get("skills", {})
            
            for skill in skills:
                all_skills.add(skill)
                skill_frequencies[skill] += 1
        
        # Identify core skills (appear in 50%+ of matches)
        core_skills = [skill for skill, freq in skill_frequencies.items() 
                      if freq >= len(matches) * 0.5]
        
        # Identify differentiating skills
        differentiating_skills = [skill for skill, freq in skill_frequencies.items() 
                                if freq == 1]
        
        return {
            "core_skills": core_skills,
            "differentiating_skills": differentiating_skills,
            "skill_overlap": len(core_skills) / len(all_skills) if all_skills else 0
        }

    def _analyze_progression_paths(self, matches: List[Dict]) -> List[Dict]:
        """Analyze potential career progression paths"""
        paths = []
        
        # Group by O*NET clusters
        cluster_groups = defaultdict(list)
        for match in matches:
            onet_code = match.get("onet_code", "")
            if onet_code:
                cluster = onet_code[:2]
                cluster_groups[cluster].append(match)
        
        for cluster, cluster_matches in cluster_groups.items():
            if len(cluster_matches) > 1:
                # Sort by score to suggest progression
                cluster_matches.sort(key=lambda x: x.get("enhanced_score", 0))
                
                paths.append({
                    "cluster": cluster,
                    "entry_role": cluster_matches[0]["job_name"],
                    "target_role": cluster_matches[-1]["job_name"],
                    "progression_steps": [m["job_name"] for m in cluster_matches]
                })
        
        return paths

    async def _generate_strategic_recommendations(self, profile: PersonProfile, matches: List[Dict]) -> Dict:
        """Generate strategic career recommendations"""
        if not matches:
            return {}
        
        # Identify skill development priorities
        skill_gaps = []
        for match in matches[:3]:  # Top 3 matches
            job_name = match["job_name"]
            scoring_details = match.get("scoring_details", {})
            skills_result = scoring_details.get("skills", {})
            gaps = skills_result.get("gaps", [])
            
            for gap in gaps[:2]:  # Top 2 gaps per job
                skill_gaps.append({
                    "skill": gap["skill"],
                    "gap": gap["gap"],
                    "job": job_name
                })
        
        # Prioritize skills that appear in multiple top jobs
        skill_priority = defaultdict(list)
        for gap in skill_gaps:
            skill_priority[gap["skill"]].append(gap)
        
        priority_skills = []
        for skill, gaps in skill_priority.items():
            if len(gaps) >= 2:  # Appears in multiple jobs
                avg_gap = sum(g["gap"] for g in gaps) / len(gaps)
                priority_skills.append({
                    "skill": skill,
                    "priority": "High",
                    "average_gap": avg_gap,
                    "relevant_jobs": [g["job"] for g in gaps]
                })
        
        priority_skills.sort(key=lambda x: x["average_gap"], reverse=True)
        
        return {
            "immediate_actions": priority_skills[:3],
            "long_term_development": priority_skills[3:6],
            "exploration_suggestions": await self._generate_exploration_suggestions(profile, matches)
        }

    async def _generate_exploration_suggestions(self, profile: PersonProfile, matches: List[Dict]) -> List[str]:
        """Generate suggestions for career exploration"""
        suggestions = []
        
        # Based on top matches, suggest informational interviews
        if matches:
            top_job = matches[0]["job_name"]
            suggestions.append(f"Schedule informational interviews with professionals in {top_job}")
        
        # Suggest skill development based on gaps
        common_gaps = self._get_most_common_gaps(matches)
        if common_gaps:
            top_gap = common_gaps[0]
            suggestions.append(f"Take an online course or bootcamp in {top_gap}")
        
        # Suggest networking in relevant industries
        industries = self._extract_industries_from_matches(matches)
        if industries:
            suggestions.append(f"Join professional associations in {industries[0]}")
        
        return suggestions[:5]  # Limit to 5 suggestions

    # ---------- Utility Methods ----------
    def _get_most_common_gaps(self, matches: List[Dict]) -> List[str]:
        """Get most common skill gaps across matches"""
        gap_counter = defaultdict(int)
        
        for match in matches:
            scoring_details = match.get("scoring_details", {})
            skills_result = scoring_details.get("skills", {})
            gaps = skills_result.get("gaps", [])
            
            for gap in gaps:
                gap_counter[gap["skill"]] += 1
        
        return [skill for skill, count in sorted(gap_counter.items(), key=lambda x: x[1], reverse=True)]

    def _extract_industries_from_matches(self, matches: List[Dict]) -> List[str]:
        """Extract industries from job matches using O*NET codes"""
        industry_map = {
            "11": "Management",
            "13": "Business and Financial Operations",
            "15": "Computer and Mathematical",
            "17": "Architecture and Engineering",
            "19": "Life, Physical, and Social Science",
            "21": "Community and Social Service",
            "23": "Legal",
            "25": "Education",
            "27": "Arts, Design, Entertainment, Sports, and Media",
            "29": "Healthcare Practitioners and Technical",
            "31": "Healthcare Support",
            "33": "Protective Service",
            "35": "Food Preparation and Serving",
            "37": "Building and Grounds Cleaning and Maintenance",
            "39": "Personal Care and Service",
            "41": "Sales and Related",
            "43": "Office and Administrative Support",
            "45": "Farming, Fishing, and Forestry",
            "47": "Construction and Extraction",
            "49": "Installation, Maintenance, and Repair",
            "51": "Production",
            "53": "Transportation and Material Moving"
        }
        
        industry_counter = defaultdict(int)
        for match in matches:
            onet_code = match.get("onet_code", "")
            if onet_code and len(onet_code) >= 2:
                prefix = onet_code[:2]
                if prefix in industry_map:
                    industry_counter[industry_map[prefix]] += 1
        
        return [industry for industry, count in sorted(industry_counter.items(), key=lambda x: x[1], reverse=True)]

    def _calculate_match_diversity(self, matches: List[Dict]) -> float:
        """Calculate how diverse the top matches are (different industries/skill sets)"""
        if len(matches) < 2:
            return 0.0
        
        # Check O*NET code diversity
        onet_prefixes = set()
        for match in matches:
            onet_code = match.get("onet_code", "")
            if onet_code and len(onet_code) >= 2:
                onet_prefixes.add(onet_code[:2])
        
        return len(onet_prefixes) / len(matches)

    # ---------- Enhanced AI Insights ----------
    async def generate_enhanced_ai_insights(self, profile: PersonProfile, job_name: str, 
                                          traditional_match: Dict, scoring_details: Dict) -> Dict:
        """Generate enhanced AI insights using detailed scoring information"""
        job_data = self.onet_jobs[job_name]
        
        # Generate all AI components with enhanced context
        ai_summary = await self._generate_enhanced_ai_summary(profile, job_name, job_data, traditional_match, scoring_details)
        keywords = await self._generate_keywords(job_name, job_data)
        onet_categories = await self._generate_onet_categories(job_name, job_data)
        action_plan = await self._generate_enhanced_action_plan(profile, job_name, traditional_match, scoring_details)
        career_story = await self._generate_career_story(profile, job_name, traditional_match)
        interview_insights = await self._generate_enhanced_interview_insights(profile, job_name, traditional_match, scoring_details)
        growth_potential = await self._generate_growth_potential_analysis(profile, job_name, job_data)
        skill_development_roadmap = await self._generate_skill_development_roadmap(profile, job_name, scoring_details)
        
        return {
            "ai_summary": ai_summary,
            "keywords": keywords,
            "onet_categories": onet_categories,
            "action_plan": action_plan,
            "career_story": career_story,
            "interview_insights": interview_insights,
            "growth_potential": growth_potential,
            "skill_development_roadmap": skill_development_roadmap,
            "similar_roles": job_data.get("similar_roles", []),
        }

    async def _generate_enhanced_ai_summary(self, profile: PersonProfile, job_name: str, 
                                          job_data: Dict, match: Dict, scoring_details: Dict) -> str:
        """Generate enhanced AI summary using detailed scoring"""
        skills_result = scoring_details.get("skills", {})
        values_result = scoring_details.get("values", {})
        
        strengths = skills_result.get("strengths", [])[:2]
        gaps = skills_result.get("gaps", [])[:2]
        value_alignments = values_result.get("alignments", [])[:2]
        
        strength_text = ", ".join([s["skill"] for s in strengths]) if strengths else "core competencies"
        gap_text = ", ".join([g["skill"] for g in gaps]) if gaps else "minor skill areas"
        alignment_text = ", ".join([a["value"] for a in value_alignments]) if value_alignments else "work values"
        
        prompt = f"""
Create a personalized 2-3 paragraph summary for {profile.name} regarding their fit for {job_name}.

Match Details:
- Overall: {match['overall_match']}%
- Key Strengths: {strength_text}
- Development Areas: {gap_text}
- Value Alignment: {alignment_text}

Tone: encouraging, professional, specific. Address both strengths and growth opportunities.
"""
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"AI call failed for {job_name}: {str(e)}")
            return (
                f"Based on your {match['overall_match']}% match with {job_name}, you demonstrate strong potential "
                f"with notable strengths in {strength_text}. Consider developing {gap_text} to increase your competitiveness."
            )

    async def _generate_enhanced_action_plan(self, profile: PersonProfile, job_name: str, 
                                           match: Dict, scoring_details: Dict) -> Dict:
        """Generate enhanced action plan using detailed gap analysis"""
        skills_result = scoring_details.get("skills", {})
        gaps = skills_result.get("gaps", [])[:3]
        
        if not gaps:
            return {
                "top_needs": ["Maintain current skill levels", "Gain industry experience", "Build professional network"],
                "action_items": [
                    "Apply to relevant positions",
                    "Join professional associations",
                    "Attend industry events",
                    "Update portfolio/resume",
                    "Practice interviewing"
                ]
            }
        
        gap_details = "\n".join([
            f"- {gap['skill']}: Current {gap['user_level']:.1f}, Need {gap['required_level']:.1f} (Gap: {gap['gap']:.1f})"
            for gap in gaps
        ])
        
        prompt = f"""
Create a focused action plan for {profile.name} to become competitive for {job_name}.

Specific skill gaps identified:
{gap_details}

Provide JSON with:
- "top_needs" (3 most critical development areas)
- "action_items" (5 specific, actionable steps)
- "timeline" (realistic timeframe for each top need)

Focus on practical, achievable steps.
"""
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.6,
            )
            content = response.choices[0].message.content.strip()
            
            import re
            match_obj = re.search(r"\{.*\}", content, re.DOTALL)
            if match_obj:
                return json.loads(match_obj.group())
        except Exception:
            pass
        
        # Fallback based on gaps
        return {
            "top_needs": [f"Develop {gap['skill']}" for gap in gaps],
            "action_items": [
                f"Take online course in {gaps[0]['skill']}" if gaps else "Build relevant skills",
                "Complete practice projects",
                "Seek mentorship or coaching",
                "Join relevant communities",
                "Document learning progress"
            ],
            "timeline": {
                gaps[0]['skill'] if gaps else "Core skills": "2-3 months",
                gaps[1]['skill'] if len(gaps) > 1 else "Experience": "3-6 months",
                gaps[2]['skill'] if len(gaps) > 2 else "Networking": "1-2 months"
            }
        }

    async def _generate_enhanced_interview_insights(self, profile: PersonProfile, job_name: str, 
                                                  match: Dict, scoring_details: Dict) -> Dict:
        """Generate enhanced interview insights using detailed match data"""
        skills_result = scoring_details.get("skills", {})
        values_result = scoring_details.get("values", {})
        
        strengths = skills_result.get("strengths", [])[:3]
        alignments = values_result.get("alignments", [])[:2]
        gaps = skills_result.get("gaps", [])[:2]
        
        selling_points = []
        if strengths:
            selling_points.extend([f"Strong {s['skill']} capabilities" for s in strengths])
        if alignments:
            selling_points.extend([f"Alignment with {a['value']} values" for a in alignments])
        
        # Pad with defaults if needed
        while len(selling_points) < 3:
            selling_points.append("Demonstrated problem-solving abilities")
        
        gap_areas = [g['skill'] for g in gaps] if gaps else []
        
        prompt = f"""
Create JSON interview prep for {profile.name} applying for {job_name}:

Strengths to highlight: {', '.join(selling_points)}
Areas to address carefully: {', '.join(gap_areas)}

Include:
- "key_selling_points" (3 specific strengths)
- "story_examples" (3 STAR method examples)
- "questions_to_ask" (3 thoughtful questions)
- "gap_mitigation" (how to address weakness questions)
"""
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=450,
                temperature=0.6,
            )
            
            import re
            match_obj = re.search(r"\{.*\}", response.choices[0].message.content.strip(), re.DOTALL)
            if match_obj:
                return json.loads(match_obj.group())
        except Exception as e:
            logger.error(f"AI call failed for {job_name}: {str(e)}")
            pass
        
        return {
            "key_selling_points": selling_points[:3],
            "story_examples": [
                "Describe a challenging project you completed successfully",
                "Share an example of learning a new skill quickly",
                "Discuss a time you collaborated effectively with others"
            ],
            "questions_to_ask": [
                "What does success look like in this role after 6 months?",
                "What are the biggest challenges the team is currently facing?",
                "How does this position contribute to the company's strategic goals?"
            ],
            "gap_mitigation": f"Emphasize learning agility and provide examples of quickly acquiring new skills like {gap_areas[0] if gap_areas else 'relevant competencies'}"
        }

    async def _generate_growth_potential_analysis(self, profile: PersonProfile, job_name: str, job_data: Dict) -> Dict:
        """Analyze growth potential and career progression"""
        similar_roles = job_data.get("similar_roles", [])
        onet_code = job_data.get("onet_code", "")
        
        # Find potential career progression within same cluster
        cluster_jobs = []
        if onet_code and len(onet_code) >= 2:
            cluster = onet_code[:2]
            cluster_jobs = self.career_clusters.get(cluster, [])
        
        prompt = f"""
Analyze growth potential for {job_name}:

Similar roles: {', '.join(similar_roles[:5])}
Career cluster: {', '.join(cluster_jobs[:5])}

Provide JSON with:
- "advancement_timeline" (typical progression timeframe)
- "next_level_roles" (2-3 natural next steps)
- "skill_evolution" (how skills will grow)
- "earning_potential" (growth trajectory)
"""
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.6,
            )
            
            import re
            match_obj = re.search(r"\{.*\}", response.choices[0].message.content.strip(), re.DOTALL)
            if match_obj:
                return json.loads(match_obj.group())
        except Exception:
            pass
        
        return {
            "advancement_timeline": "2-5 years to next level",
            "next_level_roles": similar_roles[:3] if similar_roles else ["Senior roles", "Management positions"],
            "skill_evolution": "Progressive development of expertise and leadership capabilities",
            "earning_potential": "Strong growth potential with experience and skill development"
        }

    async def _generate_skill_development_roadmap(self, profile: PersonProfile, job_name: str, scoring_details: Dict) -> Dict:
        """Generate detailed skill development roadmap"""
        skills_result = scoring_details.get("skills", {})
        gaps = skills_result.get("gaps", [])
        strengths = skills_result.get("strengths", [])
        
        if not gaps:
            return {
                "immediate": ["Maintain current skill levels"],
                "short_term": ["Gain specialized experience"],
                "long_term": ["Develop leadership capabilities"]
            }
        
        # Categorize gaps by severity and timeframe
        critical_gaps = [g for g in gaps if g.get("severity") == "Critical"]
        high_gaps = [g for g in gaps if g.get("severity") == "High"]
        medium_gaps = [g for g in gaps if g.get("severity") == "Medium"]
        
        return {
            "immediate": [f"Address {g['skill']} gap (Critical)" for g in critical_gaps[:2]],
            "short_term": [f"Develop {g['skill']} skills" for g in high_gaps[:2]],
            "long_term": [f"Enhance {g['skill']} expertise" for g in medium_gaps[:2]],
            "maintenance": [f"Maintain strength in {s['skill']}" for s in strengths[:2]]
        }

    async def _generate_ai_comparison(self, profile: PersonProfile, matches: List[Dict]) -> str:
        """Generate AI-powered comparison of top career matches"""
        if len(matches) < 2:
            return "Insufficient matches for comparison analysis."
        
        top_jobs = [m["job_name"] for m in matches[:3]]
        scores = [f"{m['job_name']}: {m.get('enhanced_score', m['overall_match'])}%" for m in matches[:3]]
        
        prompt = f"""
Provide decision framework for {profile.name} choosing between these career options:

{', '.join(scores)}

Consider:
- Skill development opportunities
- Career progression potential
- Market demand
- Personal fit

Give 2-3 practical decision criteria in 150 words.
"""
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return (
                f"When choosing between {', '.join(top_jobs)}, consider: "
                "1) Which role offers the best skill development for your long-term goals, "
                "2) Market demand and growth potential in each field, "
                "3) Cultural fit and work environment preferences."
            )

    # ---------- Backward Compatibility ----------
    def calculate_job_match(self, profile: PersonProfile, job_name: str) -> Dict:
        """Maintain backward compatibility with existing interface"""
        job_data = self.onet_jobs.get(job_name, {})
        if not job_data:
            return {}
        
        # Use original scoring methods for compatibility
        skills_score = self._calculate_skills_match(profile.skills, job_data.get("skills", {}))
        values_score = self._calculate_values_match(profile.work_values, job_data.get("work_values", {}))
        interests_score = self._calculate_interests_match(profile.interests, job_data.get("interests", {}))
        work_styles_score = self._calculate_work_styles_match(profile.personality, job_data.get("work_styles", {}))
        
        overall = (skills_score * 0.3 + values_score * 0.25 + interests_score * 0.2 + work_styles_score * 0.25) * 100
        
        strengths, improvements = self._identify_strengths_improvements(profile, job_data)
        
        return {
            "job_name": job_name,
            "onet_code": job_data.get("onet_code", ""),
            "overall_match": round(overall, 1),
            "breakdown": {
                "skills_match": round(skills_score * 100, 1),
                "values_match": round(values_score * 100, 1),
                "interests_match": round(interests_score * 100, 1),
                "work_styles_match": round(work_styles_score * 100, 1),
            },
            "strengths": strengths,
            "improvements": improvements,
            "required_skills": job_data.get("required_skills", []),
        }

    # ---------- Original Methods for Compatibility ----------
    def _calculate_skills_match(self, user_skills: Dict, job_skills: Dict) -> float:
        """Original skills matching method for backward compatibility"""
        if not job_skills:
            return 0.5
        
        scores, weights = [], []
        for skill, importance in job_skills.items():
            user_level = user_skills.get(skill, 2.5) / 5.0
            weight = importance / 5.0
            score = 1.0 if user_level >= weight else max(0, 1 - ((weight - user_level) * 2))
            scores.append(score * weight)
            weights.append(importance)
        
        return (sum(scores) / sum(weights) * 5) if scores else 0

    def _calculate_values_match(self, user_values: Dict, job_values: Dict) -> float:
        """Original values matching method"""
        if not job_values:
            return 0.5
        
        scores, weights = [], []
        for key, job_importance in job_values.items():
            user_importance = user_values.get(key, 3.5)
            normalized_job = job_importance / 5.0
            normalized_user = user_importance / 6.0
            similarity = 1 - abs(normalized_job - normalized_user)
            scores.append(similarity * normalized_job)
            weights.append(job_importance)
        
        return (sum(scores) / sum(weights) * 5) if scores else 0

    def _calculate_interests_match(self, user_interests: List[str], job_interests: Dict) -> float:
        """Original interests matching method"""
        if not user_interests or not job_interests:
            return 0.5
        
        max_possible = sum(job_interests.values())
        achieved = sum(value for key, value in job_interests.items() if key in user_interests)
        return (achieved / max_possible) if max_possible else 0

    def _calculate_work_styles_match(self, personality: Dict, job_styles: Dict) -> float:
        """Original work styles matching method"""
        if not job_styles:
            return 0.5
        
        personality_mapping = {
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
        for style, importance in job_styles.items():
            user_level = personality_mapping.get(style, 3) / 5.0
            weight = importance / 5.0
            threshold = weight * 0.8
            score = 1.0 if user_level >= threshold else max(0, 1 - ((threshold - user_level) * 2))
            scores.append(score * weight)
            weights.append(importance)
        
        return (sum(scores) / sum(weights) * 5) if scores else 0

    def _identify_strengths_improvements(self, profile: PersonProfile, job_data: Dict) -> Tuple[List[str], List[Dict]]:
        """Original strengths/improvements identification"""
        strengths, improvements = [], []
        
        # Enhanced skill mapping
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
            "Research": "research",
            "Negotiation": "negotiation",
            "innovation": "innovation",
            "innovation": "innovation",
            "Languages": "languages",
            "Artistic": "artistic",
            "Hands-On Building": "hands_on_building",
            "Writing": "writing",
            "Teamwork": "teamwork",
        }
        
        for skill_name in job_data.get("required_skills", []):
            key = skill_mapping.get(skill_name, skill_name.lower().replace(" ", "_").replace("-", "_"))
            user_level = profile.skills.get(key, 2.5)
            required_level = job_data.get("skills", {}).get(key, 3.5)
            
            if user_level >= required_level:
                strengths.append(f"Strong {skill_name} skills (Level {user_level}/5)")
            elif user_level < required_level - 0.5:
                improvements.append({
                    "skill": skill_name,
                    "current_level": user_level,
                    "required_level": required_level,
                    "gap_severity": "High" if (required_level - user_level) > 1.5 else "Medium",
                })
        
        return strengths, improvements

    # ---------- Additional Utility Methods ----------
    async def _generate_keywords(self, job_name: str, job_data: Dict) -> List[str]:
        """Generate keywords (unchanged from original)"""
        if "job_keywords" in job_data:
            return job_data["job_keywords"][:4]
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Return 4 concise keywords for {job_name}, comma-separated."}],
                max_tokens=50,
                temperature=0.5,
            )
            return [k.strip() for k in response.choices[0].message.content.strip().split(",")][:4]
        except Exception:
            return ["professional", "skilled", "dedicated", "growth-oriented"]

    async def _generate_onet_categories(self, job_name: str, job_data: Dict) -> Dict[str, List[str]]:
        """Generate O*NET categories (unchanged from original)"""
        categories = {
            "skills": list(job_data.get("skills", {}).keys())[:3],
            "work_values": list(job_data.get("work_values", {}).keys())[:3],
            "work_styles": list(job_data.get("work_styles", {}).keys())[:3],
            "interests": list(job_data.get("interests", {}).keys())[:3],
        }
        
        for key, items in categories.items():
            categories[key] = [item.replace("_", " ").title() for item in items]
        
        return categories

    async def _generate_career_story(self, profile: PersonProfile, job_name: str, match: Dict) -> str:
        """Generate career story (unchanged from original)"""
        prompt = f"""
Write a 2-paragraph first-person career story for {profile.name} targeting {job_name}.
Match score: {match['overall_match']}%. Be authentic, confident, forward-looking.
"""
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return (
                f"My journey toward {job_name} reflects my strengths that align well with this role. "
                "I'm excited to apply my skills and continue growing to make meaningful impact."
            )
    
    def get_jobs_by_zone_categories(self, profile: PersonProfile, algorithm: str = "comprehensive") -> Dict[str, List[Tuple[str, float, Dict]]]:
        """
        Get top 3 jobs from each job zone category:
        - Entry-level: Zones 1-2 (3 jobs)
        - Mid-level: Zone 3 (3 jobs)
        - Advanced: Zones 4-5 (3 jobs)
        """
        # Get all job matches first
        all_matches = self.get_top_job_matches(profile, top_n=len(self.onet_jobs), algorithm=algorithm)
        
        # Categorize by job zone
        entry_level = []  # Zones 1-2
        mid_level = []    # Zone 3
        advanced = []     # Zones 4-5
        
        for job_name, score, scoring_details in all_matches:
            job_data = self.onet_jobs.get(job_name, {})
            job_zone_data = job_data.get("job_zone")
            
            if job_zone_data is None or not isinstance(job_zone_data, dict):
                continue  # Skip jobs without proper job zone data
            
            # Extract the zone number from the nested dict
            job_zone = job_zone_data.get("zone")
            
            if job_zone is None:
                continue
            
            # Convert to int if it's a string
            try:
                job_zone = int(job_zone)
            except (ValueError, TypeError):
                continue
            
            if job_zone in [1, 2]:
                entry_level.append((job_name, score, scoring_details))
            elif job_zone == 3:
                mid_level.append((job_name, score, scoring_details))
            elif job_zone in [4, 5]:
                advanced.append((job_name, score, scoring_details))
        
        return {
            "entry_level": entry_level[:3],
            "mid_level": mid_level[:3],
            "advanced": advanced[:3]
        }

    async def analyze_person_with_zone_based_matches(self, profile: PersonProfile, algorithm: str = "comprehensive") -> Dict:
        """
        Enhanced analysis returning 9 jobs across job zone categories
        """
        zone_matches = self.get_jobs_by_zone_categories(profile, algorithm)
        
        # Flatten all matches for processing
        all_zone_matches = (
            zone_matches["entry_level"] + 
            zone_matches["mid_level"] + 
            zone_matches["advanced"]
        )
        
        enhanced_matches = {
            "entry_level": [],
            "mid_level": [],
            "advanced": []
        }
        
        # Process each category
        for category, matches in zone_matches.items():
            for job_name, score, scoring_details in matches:
                # Calculate traditional match for backward compatibility
                traditional_match = self.calculate_job_match(profile, job_name)
                
                # Generate AI insights
                ai_insights = await self.generate_enhanced_ai_insights(
                    profile, job_name, traditional_match, scoring_details
                )
                
                job_data = self.onet_jobs.get(job_name, {})
                
                enhanced_match = {
                    **traditional_match,
                    "enhanced_score": round(score * 100, 1),
                    "algorithm_used": algorithm,
                    "scoring_details": scoring_details,
                    "job_zone": job_data.get("job_zone", {}).get("zone"),  # Extract zone number
                    "job_zone_description": self._get_job_zone_description(job_data.get("job_zone")),
                    **ai_insights
                }
                
                enhanced_matches[category].append(enhanced_match)
        
        # Generate comparative analysis across all 9 jobs
        all_matches_flat = (
            enhanced_matches["entry_level"] + 
            enhanced_matches["mid_level"] + 
            enhanced_matches["advanced"]
        )
        
        comparative_analysis = await self._generate_comparative_analysis(profile, all_matches_flat)
        
        return {
            "profile": asdict(profile),
            "matches_by_zone": enhanced_matches,
            "total_matches": len(all_matches_flat),
            "zone_distribution": {
                "entry_level": len(enhanced_matches["entry_level"]),
                "mid_level": len(enhanced_matches["mid_level"]),
                "advanced": len(enhanced_matches["advanced"])
            },
            "comparative_analysis": comparative_analysis,
            "algorithm_used": algorithm,
            "total_jobs_considered": len(self.onet_jobs),
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "recommendations": await self._generate_zone_based_recommendations(profile, enhanced_matches)
        }

    def _get_job_zone_description(self, job_zone_data: Optional[dict]) -> str:
        """Get human-readable job zone description"""
        if job_zone_data is None or not isinstance(job_zone_data, dict):
            return "Unknown"
        
        # If description exists in the data, use it
        if "description" in job_zone_data:
            return job_zone_data["description"]
        
        # Otherwise, fall back to zone number lookup
        zone = job_zone_data.get("zone")
        descriptions = {
            1: "Little or no preparation needed",
            2: "Some preparation needed",
            3: "Medium preparation needed",
            4: "Considerable preparation needed",
            5: "Extensive preparation needed"
        }
        return descriptions.get(zone, "Unknown")

    async def _generate_zone_based_recommendations(self, profile: PersonProfile, zone_matches: Dict) -> Dict:
        """Generate recommendations considering job zone progression"""
        recommendations = {
            "immediate_opportunities": [],
            "short_term_goals": [],
            "long_term_aspirations": [],
            "skill_development_path": []
        }
        
        # Immediate: Best entry/mid-level match
        if zone_matches["entry_level"]:
            best_entry = zone_matches["entry_level"][0]
            recommendations["immediate_opportunities"].append(
                f"Start with {best_entry['job_name']} ({best_entry['enhanced_score']}% match)"
            )
        
        if zone_matches["mid_level"]:
            best_mid = zone_matches["mid_level"][0]
            recommendations["short_term_goals"].append(
                f"Progress to {best_mid['job_name']} ({best_mid['enhanced_score']}% match)"
            )
        
        if zone_matches["advanced"]:
            best_advanced = zone_matches["advanced"][0]
            recommendations["long_term_aspirations"].append(
                f"Aim for {best_advanced['job_name']} ({best_advanced['enhanced_score']}% match)"
            )
        
        # Build skill development path
        all_gaps = []
        for category in ["entry_level", "mid_level", "advanced"]:
            for match in zone_matches[category][:1]:  # Top match per category
                scoring_details = match.get("scoring_details", {})
                skills_result = scoring_details.get("skills", {})
                gaps = skills_result.get("gaps", [])
                for gap in gaps[:2]:
                    all_gaps.append({
                        "skill": gap["skill"],
                        "zone": category,
                        "gap": gap["gap"]
                    })
        
        # Deduplicate and prioritize
        seen_skills = set()
        for gap in sorted(all_gaps, key=lambda x: x["gap"], reverse=True):
            if gap["skill"] not in seen_skills:
                recommendations["skill_development_path"].append(
                    f"Develop {gap['skill']} (needed for {gap['zone'].replace('_', ' ')} roles)"
                )
                seen_skills.add(gap["skill"])
                if len(recommendations["skill_development_path"]) >= 5:
                    break
        
        return recommendations