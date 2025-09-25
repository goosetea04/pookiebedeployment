from fastapi import APIRouter, HTTPException
from datetime import datetime
from models.schemas import PersonProfileRequest, AnalysisResponse
from services.career_matcher import AICareerMatcher
from core.firebase_client import get_db
from firebase_admin import firestore


router = APIRouter()

# Singleton matcher (simple)
matcher = AICareerMatcher()

@router.post("/analyze-profile-top3", response_model=AnalysisResponse)
async def analyze_profile_top_3_matches(request: PersonProfileRequest):
    # Firestore write (same as main.py)
    try:
        db = get_db()
        data = request.model_dump(mode="json", exclude_none=True)
        data["created_at"] = firestore.SERVER_TIMESTAMP
        db.collection("user").document().set(data, merge=True)
    except Exception as e:
        # Do not fail the whole request if logging fails; log and continue
        print(f"[Firestore] write failed: {e}")

    try:
        profile = matcher.create_profile_from_request(request)
        result = await matcher.analyze_person_with_top_matches(profile, top_n=3)
        return AnalysisResponse(
            success=True,
            message=f"Successfully analyzed top 3 career matches for {profile.name} with AI insights",
            result=result,
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing profile: {str(e)}")

@router.post("/analyze-profile-ai", response_model=AnalysisResponse)
async def analyze_profile_with_ai(request: PersonProfileRequest):
    """Legacy: analyze all jobs."""
    try:
        profile = matcher.create_profile_from_request(request)
        matches = []
        for job_name in matcher.onet_jobs.keys():
            match_result = matcher.calculate_job_match(profile, job_name)
            ai_insights = await matcher.generate_ai_insights(profile, job_name, match_result)
            matches.append({**match_result, **ai_insights})
        matches.sort(key=lambda x: x["overall_match"], reverse=True)
        result = {
            "profile": profile.__dict__,
            "matches": matches,
            "top_match": matches[0] if matches else None,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        return AnalysisResponse(
            success=True,
            message=f"Successfully analyzed profile for {profile.name} with AI insights (all {len(matches)} jobs)",
            result=result,
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing profile: {str(e)}")

@router.post("/generate-job-insights")
async def generate_specific_job_insights(job_name: str, request: PersonProfileRequest):
    try:
        profile = matcher.create_profile_from_request(request)
        match_data = matcher.calculate_job_match(profile, job_name)
        ai_insights = await matcher.generate_ai_insights(profile, job_name, match_data)
        return {
            "success": True,
            "job_name": job_name,
            "match_data": match_data,
            "ai_insights": ai_insights,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating job insights: {str(e)}")
