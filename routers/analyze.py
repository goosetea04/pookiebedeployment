from fastapi import APIRouter, HTTPException
from datetime import datetime
from models.schemas import PersonProfileRequest, AnalysisResponse
from services.career_matcher import AICareerMatcher
from core.firebase_client import get_db
from firebase_admin import firestore


router = APIRouter()

# Singleton matcher (simple)
matcher = AICareerMatcher()

@router.get("/")
def read_root():
    return {"status": "API is running", "message": "Welcome to Career Matching API"}

@router.get("/health")
def health_check():
    return {"status": "healthy"}

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
    
@router.post("/analyze-profile-entry-level", response_model=AnalysisResponse)
async def analyze_profile_entry_level(request: PersonProfileRequest):
    """
    Analyze profile and return top 3 entry-level career matches (job zones 1-2)
    """
    # Firestore write
    try:
        db = get_db()
        data = request.model_dump(mode="json", exclude_none=True)
        data["created_at"] = firestore.SERVER_TIMESTAMP
        data["analysis_type"] = "entry_level"
        db.collection("user").document().set(data, merge=True)
    except Exception as e:
        print(f"[Firestore] write failed: {e}")

    try:
        profile = matcher.create_profile_from_request(request)
        
        # Get jobs categorized by zones
        zone_matches = matcher.get_jobs_by_zone_categories(profile, algorithm="comprehensive")
        entry_level_matches = zone_matches["entry_level"]
        
        # Process matches with AI insights
        enhanced_matches = []
        for job_name, score, scoring_details in entry_level_matches:
            traditional_match = matcher.calculate_job_match(profile, job_name)
            ai_insights = await matcher.generate_enhanced_ai_insights(
                profile, job_name, traditional_match, scoring_details
            )
            
            job_data = matcher.onet_jobs.get(job_name, {})
            
            enhanced_match = {
                **traditional_match,
                "enhanced_score": round(score * 100, 1),
                "algorithm_used": "comprehensive",
                "scoring_details": scoring_details,
                "job_zone": job_data.get("job_zone", {}).get("zone"),
                "job_zone_description": matcher._get_job_zone_description(job_data.get("job_zone")),
                **ai_insights
            }
            enhanced_matches.append(enhanced_match)
        
        # Sort by overall_match in descending order
        enhanced_matches.sort(key=lambda x: x["overall_match"], reverse=True)
        
        # Generate analysis specific to entry-level
        comparative_analysis = await matcher._generate_comparative_analysis(profile, enhanced_matches)
        
        result = {
            "profile": profile.__dict__,
            "job_zone_category": "entry_level",
            "job_zones_included": [1, 2],
            "matches": enhanced_matches,
            "total_matches": len(enhanced_matches),
            "comparative_analysis": comparative_analysis,
            "recommendations": {
                "focus": "Entry-level positions requiring minimal preparation",
                "next_steps": [
                    f"Apply for {enhanced_matches[0]['job_name']}" if enhanced_matches else "Explore entry-level opportunities",
                    "Build foundational skills through online courses",
                    "Seek internships or apprenticeships"
                ]
            },
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return AnalysisResponse(
            success=True,
            message=f"Successfully analyzed top 3 entry-level matches (zones 1-2) for {profile.name}",
            result=result,
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing entry-level profile: {str(e)}")


@router.post("/analyze-profile-mid-level", response_model=AnalysisResponse)
async def analyze_profile_mid_level(request: PersonProfileRequest):
    """
    Analyze profile and return top 3 mid-level career matches (job zone 3)
    """
    # Firestore write
    try:
        db = get_db()
        data = request.model_dump(mode="json", exclude_none=True)
        data["created_at"] = firestore.SERVER_TIMESTAMP
        data["analysis_type"] = "mid_level"
        db.collection("user").document().set(data, merge=True)
    except Exception as e:
        print(f"[Firestore] write failed: {e}")

    try:
        profile = matcher.create_profile_from_request(request)
        
        # Get jobs categorized by zones
        zone_matches = matcher.get_jobs_by_zone_categories(profile, algorithm="comprehensive")
        mid_level_matches = zone_matches["mid_level"]
        
        # Process matches with AI insights
        enhanced_matches = []
        for job_name, score, scoring_details in mid_level_matches:
            traditional_match = matcher.calculate_job_match(profile, job_name)
            ai_insights = await matcher.generate_enhanced_ai_insights(
                profile, job_name, traditional_match, scoring_details
            )
            
            job_data = matcher.onet_jobs.get(job_name, {})
            
            enhanced_match = {
                **traditional_match,
                "enhanced_score": round(score * 100, 1),
                "algorithm_used": "comprehensive",
                "scoring_details": scoring_details,
                "job_zone": job_data.get("job_zone", {}).get("zone"),
                "job_zone_description": matcher._get_job_zone_description(job_data.get("job_zone")),
                **ai_insights
            }
            enhanced_matches.append(enhanced_match)
        
        # Sort by overall_match in descending order
        enhanced_matches.sort(key=lambda x: x["overall_match"], reverse=True)
        
        # Generate analysis specific to mid-level
        comparative_analysis = await matcher._generate_comparative_analysis(profile, enhanced_matches)
        
        result = {
            "profile": profile.__dict__,
            "job_zone_category": "mid_level",
            "job_zones_included": [3],
            "matches": enhanced_matches,
            "total_matches": len(enhanced_matches),
            "comparative_analysis": comparative_analysis,
            "recommendations": {
                "focus": "Mid-level positions requiring medium preparation",
                "next_steps": [
                    f"Target {enhanced_matches[0]['job_name']} roles" if enhanced_matches else "Build mid-level competencies",
                    "Obtain relevant certifications or vocational training",
                    "Gain 1-2 years of relevant experience"
                ]
            },
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return AnalysisResponse(
            success=True,
            message=f"Successfully analyzed top 3 mid-level matches (zone 3) for {profile.name}",
            result=result,
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing mid-level profile: {str(e)}")


@router.post("/analyze-profile-advanced", response_model=AnalysisResponse)
async def analyze_profile_advanced(request: PersonProfileRequest):
    """
    Analyze profile and return top 3 advanced career matches (job zones 4-5)
    """
    # Firestore write
    try:
        db = get_db()
        data = request.model_dump(mode="json", exclude_none=True)
        data["created_at"] = firestore.SERVER_TIMESTAMP
        data["analysis_type"] = "advanced"
        db.collection("user").document().set(data, merge=True)
    except Exception as e:
        print(f"[Firestore] write failed: {e}")

    try:
        profile = matcher.create_profile_from_request(request)
        
        # Get jobs categorized by zones
        zone_matches = matcher.get_jobs_by_zone_categories(profile, algorithm="comprehensive")
        advanced_matches = zone_matches["advanced"]
        
        # Process matches with AI insights
        enhanced_matches = []
        for job_name, score, scoring_details in advanced_matches:
            traditional_match = matcher.calculate_job_match(profile, job_name)
            ai_insights = await matcher.generate_enhanced_ai_insights(
                profile, job_name, traditional_match, scoring_details
            )
            
            job_data = matcher.onet_jobs.get(job_name, {})
            
            enhanced_match = {
                **traditional_match,
                "enhanced_score": round(score * 100, 1),
                "algorithm_used": "comprehensive",
                "scoring_details": scoring_details,
                "job_zone": job_data.get("job_zone", {}).get("zone"),
                "job_zone_description": matcher._get_job_zone_description(job_data.get("job_zone")),
                **ai_insights
            }
            enhanced_matches.append(enhanced_match)
        
        # Sort by overall_match in descending order
        enhanced_matches.sort(key=lambda x: x["overall_match"], reverse=True)
        
        # Generate analysis specific to advanced level
        comparative_analysis = await matcher._generate_comparative_analysis(profile, enhanced_matches)
        
        result = {
            "profile": profile.__dict__,
            "job_zone_category": "advanced",
            "job_zones_included": [4, 5],
            "matches": enhanced_matches,
            "total_matches": len(enhanced_matches),
            "comparative_analysis": comparative_analysis,
            "recommendations": {
                "focus": "Advanced positions requiring considerable to extensive preparation",
                "next_steps": [
                    f"Pursue advanced education for {enhanced_matches[0]['job_name']}" if enhanced_matches else "Consider graduate education",
                    "Develop specialized expertise in your field",
                    "Build leadership and strategic thinking capabilities"
                ]
            },
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return AnalysisResponse(
            success=True,
            message=f"Successfully analyzed top 3 advanced matches (zones 4-5) for {profile.name}",
            result=result,
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing advanced profile: {str(e)}")

@router.post("/analyze-profile-by-zones", response_model=AnalysisResponse)
async def analyze_profile_by_job_zones(request: PersonProfileRequest):
    """
    Analyze profile and return 9 career matches:
    - 3 from job zones 1-2 (entry level)
    - 3 from job zone 3 (mid level)
    - 3 from job zones 4-5 (advanced)
    """
    # Firestore write
    try:
        db = get_db()
        data = request.model_dump(mode="json", exclude_none=True)
        data["created_at"] = firestore.SERVER_TIMESTAMP
        db.collection("user").document().set(data, merge=True)
    except Exception as e:
        print(f"[Firestore] write failed: {e}")

    try:
        profile = matcher.create_profile_from_request(request)
        result = await matcher.analyze_person_with_zone_based_matches(
            profile, 
            algorithm="comprehensive"
        )
        
        return AnalysisResponse(
            success=True,
            message=f"Successfully analyzed 9 career matches across job zones for {profile.name}",
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
        
        # Sort by overall_match in descending order
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