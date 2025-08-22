from langchain_core.tools import tool
import json

@tool
def estimate_tdee(profile_json: str):
    """Mifflin–St Jeor 공식을 사용해 BMR과 활동계수로 TDEE를 추정합니다.
    입력은 JSON 문자열(성별/나이/키/체중/활동수준)이어야 합니다.
    """
    p = json.loads(profile_json)
    s,a,h,w = p["sex"],p["age"],p["height_cm"],p["weight_kg"]
    bmr = 10*w + 6.25*h - 5*a + (5 if s.upper().startswith("M") else -161)
    mult = {"sedentary":1.2,"light":1.375,"moderate":1.55,"high":1.725}.get(p.get("activity","moderate"),1.55)
    return {"bmr": round(bmr), "tdee": round(bmr*mult)}

@tool
def macro_plan(goal_json: str):
    """TDEE와 목표(감량/유지/증량)에 따라 일일 칼로리와 P/F/C 매크로를 계산합니다.
    입력은 JSON 문자열(체중, TDEE, goal)입니다.
    """
    g = json.loads(goal_json)
    w, tdee, goal = g["weight_kg"], g["tdee"], g["goal"]
    adj = {"cut":-0.2,"recomp":0.0,"bulk":0.1}.get(goal,0.0)
    target = int(tdee*(1+adj))
    protein = int(round(w*2.0)); fat = int(round(w*0.8))
    carbs = max(0, (target - protein*4 - fat*9)//4)
    return {"kcal":target,"protein_g":protein,"fat_g":fat,"carbs_g":carbs}

@tool
def exercise_picker(criteria_json: str):
    """근육/장비/회피운동 조건에 따라 추천 운동 목록을 반환합니다.
    입력은 JSON 문자열(muscle, equipment[], avoid[])입니다.
    """
    c = json.loads(criteria_json)
    muscle = c.get("muscle","Back")
    equipment = set(c.get("equipment",["barbell","dumbbell"]))
    avoid = set([a.lower() for a in c.get("avoid",[])])
    base = [
        {"muscle":"Chest","exercise":"Barbell Bench Press","equipment":"barbell","RIR":"2-3"},
        {"muscle":"Back","exercise":"Barbell Row","equipment":"barbell","RIR":"2-3"},
        {"muscle":"Shoulders","exercise":"Overhead Press","equipment":"barbell","RIR":"2-3"},
        {"muscle":"Legs","exercise":"Front Squat","equipment":"barbell","RIR":"2-3"},
        {"muscle":"Legs","exercise":"Romanian Deadlift","equipment":"barbell","RIR":"2-3"},
        {"muscle":"Delts","exercise":"Side Lateral Raise","equipment":"dumbbell","RIR":"1-2"},
        {"muscle":"Triceps","exercise":"Cable Triceps Extension","equipment":"cable","RIR":"1-2"},
    ]
    picks = [x for x in base if x["muscle"].lower()==muscle.lower() and x["equipment"] in equipment and x["exercise"].lower() not in avoid]
    return picks[:6] if picks else [{"note":"No match"}]

@tool
def contraindication_check(profile_json: str):
    """부상/질환 플래그에 따라 주의 및 대체운동을 제안합니다.
    입력은 JSON 문자열(conditions[])입니다.
    """
    p = json.loads(profile_json)
    conds = [x.lower() for x in p.get("conditions",[])]
    warnings = []
    if "knee pain" in conds: warnings.append("무릎: Front Squat 대신 Split Squat/Leg Press 고려")
    if "shoulder pain" in conds: warnings.append("어깨: OHP 대신 Landmine Press/Incline DB Press 고려")
    if "hypertension" in conds: warnings.append("고혈압: 고용량 카페인(>200mg) 회피/의사 상담")
    return warnings or ["None"]
