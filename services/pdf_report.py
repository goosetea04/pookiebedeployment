import io
from datetime import datetime
from typing import Dict, Tuple, List
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

class PDFReportGenerator:
    """Extracted from your main file; unchanged output structure."""
    def generate_pdf_report(self, analysis_data: Dict, job_name: str) -> io.BytesIO:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        avail_w = doc.width

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontSize=28, spaceAfter=10,
                                     textColor=colors.HexColor('#1a1a1a'), fontName='Helvetica-Bold')
        email_style = ParagraphStyle('EmailStyle', parent=styles['Normal'], fontSize=12, spaceAfter=30,
                                     textColor=colors.HexColor('#666666'))
        cell_style = ParagraphStyle('Cell', parent=styles['Normal'], fontSize=9, leading=12,
                                    textColor=colors.black, spaceAfter=4, wordWrap='CJK')
        section_header_style = ParagraphStyle('SectionHeader', parent=styles['Heading1'], fontSize=20,
                                              spaceAfter=15, textColor=colors.HexColor('#2E3440'),
                                              fontName='Helvetica-Bold')
        subsection_style = ParagraphStyle('SubsectionHeader', parent=styles['Heading2'], fontSize=16,
                                          spaceAfter=10, textColor=colors.HexColor('#5E81AC'),
                                          fontName='Helvetica-Bold')
        body_style = ParagraphStyle('BodyText', parent=styles['Normal'], fontSize=11, spaceAfter=8, leading=14)

        job_match = next((m for m in analysis_data['matches'] if m['job_name'] == job_name),
                         analysis_data['matches'][0])

        story: List = []
        story.append(Paragraph(analysis_data['profile']['name'], title_style))
        story.append(Paragraph(analysis_data['profile']['email'], email_style))

        story.append(Paragraph("Top 3 Career Matches", section_header_style))
        methodology_text = """We have identified your top career matches using a sophisticated algorithm that integrates your preferences,
personality, and skills with five proven, industry-leading frameworks and assessments:<br/><br/>
1. <b>RIASEC</b> • 2. <b>OCEAN</b> • 3. <b>Skills</b> • 4. <b>Values</b> • 5. <b>Direct Skills</b>
"""
        story.append(Paragraph(methodology_text, body_style))
        story.append(Spacer(1, 20))

        career_data = [['Career', 'Overall %', 'Skills %', 'Values %', 'Interest %', 'Personality %', '1-line Why']]
        for match in analysis_data['matches'][:3]:
            why = self._generate_one_line_why(match, analysis_data['profile'])
            career_data.append([
                Paragraph(match['job_name'], cell_style),
                f"{match['overall_match']}%",
                f"{match['breakdown']['skills_match']}%",
                f"{match['breakdown']['values_match']}%",
                f"{match['breakdown']['interests_match']}%",
                f"{match['breakdown']['work_styles_match']}%",
                Paragraph(why, cell_style)
            ])

        table = Table(career_data, colWidths=[
            0.26*avail_w, 0.10*avail_w, 0.10*avail_w, 0.10*avail_w, 0.10*avail_w, 0.10*avail_w, 0.24*avail_w
        ])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E8E8E8')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('ALIGN', (1,0), (5,-1), 'CENTER'),
            ('ALIGN', (0,0), (0,-1), 'LEFT'),
            ('ALIGN', (6,0), (6,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('FONTSIZE', (0,1), (-1,-1), 9),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
            ('VALIGN', (0,0), (-1,-1), 'TOP')
        ]))
        story.append(table)
        story.append(Spacer(1, 30))

        story.append(Paragraph(f"Most Compatible Field: {job_match['job_name']}", subsection_style))
        story.append(Paragraph(f"Score: {job_match['overall_match']}%", body_style))
        story.append(Spacer(1, 10))

        strengths = "<br/>".join([f"• {s}" for s in job_match.get('strengths', [])[:4]])
        imps = job_match.get('improvements', [])[:3]
        if imps:
            gaps = "<br/>".join([f"• {i['skill']}: improve {i['current_level']}/5 → {i['required_level']}/5" for i in imps])
        else:
            gaps = "• Strong alignment across key areas<br/>• Minor refinements may boost advancement<br/>"

        sg_table = Table([['Strengths', 'Gaps'], [Paragraph(strengths, cell_style), Paragraph(gaps, cell_style)]],
                         colWidths=[avail_w/2, avail_w/2])
        sg_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F5F5F5')),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('FONTSIZE', (0,1), (-1,-1), 10),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ]))
        story.append(sg_table)
        story.append(Spacer(1, 20))

        story.append(Paragraph("Improvement Hacks:", subsection_style))
        action_plan = job_match.get('action_plan', {})
        for item in action_plan.get('action_items', [])[:4]:
            story.append(Paragraph(f"• {item}", body_style))
        story.append(Spacer(1, 20))

        story.append(Paragraph("Interview Tips", subsection_style))
        interview = job_match.get('interview_insights', {})
        if 'key_selling_points' in interview:
            story.append(Paragraph(f"• <b>Open with:</b> \"I'm a {self._create_opening_line(job_match, analysis_data['profile'])}\"", body_style))
            for i, point in enumerate(interview['key_selling_points'][:3], 2):
                story.append(Paragraph(f"• <b>Point {i}:</b> {point}", body_style))
        if 'questions_to_ask' in interview and interview['questions_to_ask']:
            story.append(Paragraph(f"• <b>Close with a fit test:</b> \"{interview['questions_to_ask'][0]}\"", body_style))

        story.append(PageBreak())
        story.append(Paragraph("Skills", section_header_style))

        user_skills = analysis_data['profile']['skills']
        top_sk = sorted(user_skills.items(), key=lambda x: x[1], reverse=True)[:3]
        weak_sk = sorted(user_skills.items(), key=lambda x: x[1])[:2]

        works_text = f"<b>Most-Matched Skill:</b> {top_sk[0][0].replace('_',' ').title()} (Level {top_sk[0][1]}/5) — {self._get_skill_insight(top_sk[0][0])}<br/><br/>"
        works_text += f"<b>Secondary Strengths:</b> {', '.join([s.replace('_',' ').title() for s,_ in top_sk[1:3]])}"

        doesnt_text = f"<b>Largest Gap Skill:</b> {weak_sk[0][0].replace('_',' ').title()} (Level {weak_sk[0][1]}/5) — {self._get_improvement_insight(weak_sk[0][0])}<br/><br/>"
        doesnt_text += f"<b>Action:</b> Focus development on {weak_sk[0][0].replace('_',' ').lower()}."

        skills_table = Table(
            [['What Works?', 'What Doesn\'t?'],
             [Paragraph(works_text, cell_style), Paragraph(doesnt_text, cell_style)]],
            colWidths=[avail_w/2, avail_w/2]
        )
        skills_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F0F8FF')),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('FONTSIZE', (0,1), (-1,-1), 10),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
            ('BOTTOMPADDING', (0,0), (-1,-1), 15),
        ]))
        story.append(skills_table)
        story.append(Spacer(1, 25))

        story.append(Paragraph("Top 5 Industries", subsection_style))
        for i, (ind, desc) in enumerate(self._get_related_industries(job_match)[:5], 1):
            story.append(Paragraph(f"{i}. <b>{ind}</b> — {desc}", body_style))
        story.append(Spacer(1, 20))

        story.append(Paragraph("Values Alignment Check", subsection_style))
        story.append(Paragraph(self._generate_values_insight(job_match, analysis_data['profile']), body_style))

        story.append(Paragraph("Your Professional Narrative", subsection_style))
        story.append(Paragraph(job_match.get('career_story', 'Your career story...'), body_style))

        story.append(Spacer(1, 30))
        story.append(Paragraph(
            f"Report generated on {analysis_data.get('analysis_date', datetime.now().strftime('%Y-%m-%d'))}",
            ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9, textColor=colors.gray)
        ))
        doc.build(story)
        buffer.seek(0)
        return buffer

    # --- Helper text (kept from your logic) ---
    def _get_skill_insight(self, skill: str) -> str:
        return {
            'programming': 'Essential for technical roles and automation',
            'creative': 'Drives innovation and unique problem-solving approaches',
            'leadership': 'Critical for team management and project direction',
            'problem_solving': 'Core competency for analytical and strategic roles',
            'working_with_people': 'Vital for collaborative and client-facing positions',
            'math': 'Foundation for analytical and quantitative roles'
        }.get(skill, 'Valuable asset for professional success')

    def _get_improvement_insight(self, skill: str) -> str:
        return {
            'programming': 'Many modern roles expect basic coding literacy',
            'public_speaking': 'Essential for leadership and visibility',
            'networking': 'Critical for career advancement and opportunities',
            'tech_savvy': 'Increasingly important across all industries',
            'time_management': 'Fundamental for productivity and reliability'
        }.get(skill, 'Important for well-rounded professional development')

    def _generate_one_line_why(self, match: Dict, profile: Dict) -> str:
        top = sorted(profile['skills'].items(), key=lambda x: x[1], reverse=True)[:2]
        skills_text = f"{top[0][0].replace('_',' ')} + {top[1][0].replace('_',' ')}"
        p = profile['personality']
        if p.get('extraversion', 3) >= 4: note = "leadership and visibility"
        elif p.get('openness', 3) >= 4: note = "innovation and creativity"
        else: note = "analytical approach"
        return f"Perfect blend of {skills_text} skills with {note}."

    def _get_related_industries(self, job_match: Dict) -> List[Tuple[str, str]]:
        j = job_match['job_name'].lower()
        mapping = {
            'software': [('Technology & Software','High growth'),('Financial Services','FinTech'),
                         ('Healthcare Technology','Meaningful impact'),('Consulting','Digital strategy'),
                         ('Startups','Diverse challenges')],
            'marketing': [('Advertising & Media','Brand storytelling'),('Technology','Product marketing'),
                          ('Consumer Goods','Brand mgmt'),('Healthcare','Patient engagement'),
                          ('Professional Services','B2B marketing')],
            'analyst': [('Financial Services','Investment & risk'),('Consulting','Strategy'),
                        ('Technology','BI & data'),('Healthcare','Outcomes research'),('Government','Policy')],
        }
        default = [('Professional Services','Consulting'),('Technology','Innovation'),
                   ('Financial Services','Strategy'),('Healthcare','Impact'),('Education','Development')]
        for k in mapping:
            if k in j: return mapping[k]
        return default

    def _generate_values_insight(self, job_match: Dict, profile: Dict) -> str:
        ws = profile['work_values']
        top = max(ws.items(), key=lambda x: x[1])[0]
        fit = {
            'income': "This role typically offers competitive compensation with growth potential.",
            'impact': f"Your work in {job_match['job_name']} contributes to meaningful outcomes.",
            'stability': f"{job_match['job_name']} roles offer strong job security.",
            'variety': "This position provides diverse challenges and project variety.",
            'recognition': "Success here is highly visible and valued.",
            'autonomy': "This role offers independence and decision-making authority.",
        }.get(top, f"Your top value ({top}) aligns well with this path.")
        return f"<b>{top.title()} Priority:</b> {fit}"

    def _create_opening_line(self, job_match: Dict, profile: Dict) -> str:
        name = job_match['job_name'].lower()
        top = sorted(profile['skills'].items(), key=lambda x: x[1], reverse=True)[:2]
        descriptors = {
            'programming': 'technical problem-solver',
            'creative': 'innovative thinker',
            'leadership': 'results-driven leader',
            'problem_solving': 'analytical problem-solver',
            'working_with_people': 'collaborative professional',
            'math': 'quantitative analyst',
        }
        primary = descriptors.get(top[0][0], 'dedicated professional')
        if 'analyst' in name: return f"{primary} who turns complex data into actionable insights"
        if 'manager' in name or 'director' in name: return f"{primary} who drives team success"
        if 'developer' in name or 'engineer' in name: return f"{primary} who builds scalable solutions"
        return f"{primary} passionate about creating value"
