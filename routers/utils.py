from uuid import UUID
from fastapi import APIRouter, HTTPException, Request
from datetime import datetime
import io, urllib.parse, json
from fastapi.responses import StreamingResponse
from datetime import datetime, timezone
from services.career_matcher import AICareerMatcher
from services.pdf_report import PDFReportGenerator
from core.firebase_client import get_db
from core.openai_client import openai_client
from core.config import settings

router = APIRouter()

matcher = AICareerMatcher()
pdf_gen = PDFReportGenerator()

@router.get("/")
async def root():
    return {
        "message": "AI-Enhanced Career Matching API - Top 3 Focus",
        "version": settings.APP_VERSION,
        "optimization": "Now focuses on top 3 job matches for efficient AI analysis",
        "features": [
            "Top 3 job match focus (efficient)",
            "AI-powered job fit summaries",
            "Personalized action plans",
            "Interview preparation insights",
            "Career story generation",
            "PDF report downloads",
            "Quick match preview",
        ],
        "endpoints": {
            "/analyze-profile-top3": "POST - AI analysis of top 3 matches (RECOMMENDED)",
            "/analyze-profile-ai": "POST - Full AI analysis of all jobs (legacy, slower)",
            "/quick-match-preview": "GET - Quick preview without AI insights",
            "/generate-job-insights": "POST - Generate AI insights for specific job",
            "/download-report/{job_name}": "GET - Download PDF report",
            "/jobs": "GET - List available job types",
            "/health": "GET - Health check",
        },
        "efficiency_note": f"Database contains {len(matcher.onet_jobs)} jobs. Top 3 analysis reduces API calls by ~80%.",
    }

@router.get("/jobs")
async def get_jobs():
    jobs_info = {}
    for job_name, job_data in matcher.onet_jobs.items():
        jobs_info[job_name] = {
            "onet_code": job_data["onet_code"],
            "required_skills": job_data["required_skills"],
            "similar_roles": job_data.get("similar_roles", []),
            "keywords": job_data.get("job_keywords", []),
            "top_work_values": sorted(job_data["work_values"].items(), key=lambda x: x[1], reverse=True)[:3],
            "top_interests": sorted(job_data["interests"].items(), key=lambda x: x[1], reverse=True)[:3],
        }
    return {
        "total_jobs": len(matcher.onet_jobs),
        "available_jobs": list(matcher.onet_jobs.keys()),
        "job_details": jobs_info,
        "ai_features": [
            "Personalized fit analysis",
            "Career development roadmap",
            "Interview preparation guide",
            "Professional narrative development",
        ],
        "recommendation": "Use /analyze-profile-top3 for efficient analysis of best matches",
    }

@router.get("/quick-match-preview")
async def get_quick_match_preview(
    name: str,
    math: int = 3,
    programming: int = 3,
    creative: int = 3,
    working_with_people: int = 3,
    leadership: int = 3,
):
    try:
        minimal_skills = {
            "math": math, "programming": programming, "creative": creative,
            "working_with_people": working_with_people, "leadership": leadership,
            "problem_solving": 3, "tech_savvy": 3, "teamwork": 3, "attention_to_detail": 3,
            "research": 3, "writing": 3, "public_speaking": 3, "networking": 3,
            "empathy": 3, "time_management": 3, "project_management": 3,
        }
        from ..models.schemas import PersonProfile
        profile = PersonProfile(
            name=name, email="preview@example.com", university="Preview",
            personality={"openness":3,"conscientiousness":3,"extraversion":3,"agreeableness":3,"neuroticism":3},
            work_values={"income":3,"impact":3,"stability":3,"variety":3,"recognition":3,"autonomy":3},
            skills=minimal_skills, interests=["investigative"], preferred_career="Preview"
        )
        top = matcher.get_top_job_matches(profile, 3)
        quick = []
        for job_name in top:
            md = matcher.calculate_job_match(profile, job_name)
            quick.append({
                "job_name": job_name,
                "overall_match": md["overall_match"],
                "skills_match": md["breakdown"]["skills_match"],
                "required_skills": md["required_skills"][:3],
            })
        return {
            "success": True,
            "message": f"Quick match preview for {name}",
            "top_matches": quick,
            "note": "This is a preview. Use /analyze-profile-top3 for full AI analysis.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quick preview: {str(e)}")

@router.get("/download-report/{job_name}")
async def download_career_report(job_name: str, analysis_data: str):
    try:
        decoded = urllib.parse.unquote(analysis_data)
        analysis_dict = json.loads(decoded)
        pdf_buffer = pdf_gen.generate_pdf_report(analysis_dict, job_name)
        return StreamingResponse(
            io.BytesIO(pdf_buffer.read()),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=career_report_{job_name.replace(' ','_')}.pdf"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating PDF report: {str(e)}")

@router.get("/health")
async def health_check():
    ai_status = "available"
    fb_status = "available"
    try:
        _ = openai_client.chat(model="gpt-4o-mini", messages=[{"role": "user", "content": "test"}], max_tokens=1)
        ai_status = "connected"
    except Exception:
        ai_status = "unavailable"

    try:
        db = get_db()
        # cheap read to verify
        _ = db.collection("_health").document("_ping").get()
        fb_status = "connected"
    except Exception:
        fb_status = "unavailable"

    return {
        "status": "healthy" if ai_status=="connected" else "degraded",
        "ai_service": ai_status,
        "firebase": fb_status,
        "total_jobs_in_database": len(matcher.onet_jobs),
        "optimization": "Top 3 matching active",
        "timestamp": datetime.now().isoformat(),
        "version": settings.APP_VERSION,
    }

@router.post("/feedback")
async def inject_feedback(request: Request):
    try:
        # Check if request has a body
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Request body is empty")
        
        # Parse JSON
        try:
            data = await request.json()
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

        # Validate required fields
        required_fields = ["rating", "comment", "wantsUpdates"]
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"Missing field: {field}")

        # Connect to Firestore
        db = get_db()
        feedback_collection = db.collection("feedback")

        # Prepare document data
        feedback_data = {
            "rating": data.get("rating"),
            "comment": data.get("comment"),
            "wantsUpdates": data.get("wantsUpdates", False),
            "name": data.get("name") or None,
            "email": data.get("email") or None,
            "university": data.get("university") or None,
            "location": data.get("city") or None,
            "timestamp": datetime.utcnow()
        }

        # Add to Firestore
        feedback_collection.add(feedback_data)

        return {"status": "success", "message": "Feedback stored successfully."}

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print("Error saving feedback:", e)
        raise HTTPException(status_code=500, detail=f"Failed to store feedback: {str(e)}")
    
@router.post("/set-feedback")
async def set_feedback(request: Request):
    try:
        # Parse JSON
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Request body is empty")
        
        try:
            data = await request.json()
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

        # Validate required fields
        required_fields = ["email", "rating", "feedback"]
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

        # Validate rating is a number
        try:
            rating = int(data.get("rating"))
            if rating < 1 or rating > 5:
                raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="Rating must be a valid number")

        # Connect to Firestore
        db = get_db()
        feedback_collection = db.collection("feedback")

        # Prepare document data
        feedback_data = {
            "email": data.get("email"),
            "rating": rating,
            "feedback": data.get("feedback"),
            "timestamp": datetime.utcnow()
        }

        # Add to Firestore
        doc_ref = feedback_collection.add(feedback_data)

        return {
            "status": "success",
            "message": "Feedback stored successfully",
            "document_id": doc_ref[1].id
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print("Error storing feedback:", e)
        raise HTTPException(status_code=500, detail=f"Failed to store feedback: {str(e)}")