import os
import json
import time
from openai import OpenAI
from typing import Any, List, Optional

SYSTEM_PROMPT = """You are a surgical scene information normalizer.
You will be given `i_pred_raw` and MUST RETURN a STRICT JSON object with EXACTLY these keys:

{
  "quadruple": [
    ["<hand_identity>", "<instrument>", "<action>", "<target>"],
    ...
  ],
  "structures": ["<name1>", "<name2>", ...] OR null,
  "cvs": [float(0.~1.0), float(0.~1.0), float(0.~1.0)]
}

Rules:
- Use ONLY information present in i_pred_raw; do NOT invent facts.
- For each quadruple, order is [hand, instrument, action, target].
- If any element is missing/unspecified, put null at that position (do NOT drop the quadruple).
- Hand identity must be one of ["Rt oper", "Lt oper", "Assistant", "Unknown"]; if unspecified, use "Unknown".
- Normalize instrument names as short nouns where possible (e.g., "grasper", "hook", "clipper", "scissors", "suction").
- "structures": include only structures explicitly visible in i_pred_raw. Use lowercase_snake_case.
  If NONE are explicitly visible, return null.
- "cvs": always 3 floats possibility [C1,C2,C3], and they MUST be extracted exactly as given in i_pred_raw (do not infer or fabricate).

- Scene-level hand constraint (IMPORTANT):
  • Within a single i_pred_raw (scene), at most ONE of each hand identity may appear across all quadruple entries:
    - at most one "Rt oper"
    - at most one "Lt oper"
    - at most one "Assistant"
  • If the text would imply more than one quadruple with the same hand identity, KEEP the most strongly implied (or earliest) one
    and set the hand of any additional quadruples to "Unknown" (do NOT drop those quadruples).
  • Never invent or reassign actions/targets to satisfy this rule; only adjust the hand field to "Unknown" when necessary.

- Return a SINGLE JSON object ONLY. No markdown, no commentary.
"""

USER_TEMPLATE = """i_pred_raw:
<<<
{i_pred_raw}
>>>

Return the strict JSON object per schema.
"""

def build_user_prompt(i_pred_raw: str) -> str:
    return USER_TEMPLATE.format(i_pred_raw=i_pred_raw)

# ------------ 보정 함수들 ------------
def coerce_quadruple_list(raw_quadruple: Any) -> List[List[Optional[str]]]:
    fixed: List[List[Optional[str]]] = []
    if not isinstance(raw_quadruple, list):
        return fixed
    for t in raw_quadruple:
        if isinstance(t, (list, tuple)):
            row = list(t)[:4]
        elif isinstance(t, dict):
            row = [
                t.get("hand_identity"),
                t.get("instrument"),
                t.get("action"),
                t.get("target"),
            ]
        else:
            row = [None, None, None, None]

        while len(row) < 4:
            row.append(None)

        if row[0] is None or (isinstance(row[0], str) and row[0].strip() == ""):
            row[0] = "Unknown"

        # 빈 문자열을 None으로 통일
        row = [r if (isinstance(r, str) and r.strip()) else None for r in row]
        fixed.append(row)
    return fixed

def coerce_anatomical(structs: Any) -> Optional[List[str]]:
    if not isinstance(structs, list):
        return None
    cleaned = [s.strip().lower() for s in structs if isinstance(s, str) and s.strip()]
    return cleaned if cleaned else None

def coerce_cvs(cvs: Any) -> List[int]:
    out = [0, 0, 0]
    if isinstance(cvs, list):
        for i in range(min(3, len(cvs))):
            try:
                out[i] = 1 if int(cvs[i]) == 1 else 0
            except Exception:
                out[i] = 0
    return out

def normalize_to_target_schema(model_out: dict) -> dict:
    quadruple = coerce_quadruple_list(model_out.get("quadruple"))
    anatomical = coerce_anatomical(model_out.get("structures"))
    cvs = (model_out.get("cvs"))
    return {
        "quadruple": quadruple,
        "structures": anatomical,
        "cvs": cvs,
    }
# -----------------------------------

def main():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"]) # SMC
    # client = OpenAI(api_key=os.environ["OPENAI_API_KEY"]) # AMI
    json_path = "ALL_THING_no_igt.json"   # 필요시 경로 수정

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for idx, item in enumerate(data):
        if "i_pred_refined" in item:
            continue  # 이미 있으면 스킵

        i_pred_raw_text = item.get("i_pred_raw")
        if not i_pred_raw_text:
            item["i_pred_refined"] = {
                "structures": [],
                "quadruple": [],
                "cvs": [0, 0, 0],
            }
        else:
            resp = client.chat.completions.create(
                model="gpt-5-2025-08-07",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(i_pred_raw_text)},
                ],
                response_format={"type": "json_object"},
            )
            model_out = json.loads(resp.choices[0].message.content)
            item["i_pred_refined"] = normalize_to_target_schema(model_out)

        # ✅ 항목 처리 후 매번 파일에 즉시 반영
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[{idx+1}/{len(data)}] updated and saved")
        time.sleep(0.5)  # rate-limit 방지용

if __name__ == "__main__":
    main()
