import pandas as pd
from configs.paths import QUESTIONNAIRE_DIR


def load_questionnaire(subject_id):
    path = QUESTIONNAIRE_DIR / f"{subject_id}.xlsx"
    if not path.exists():
        print(f"[MISSING] Questionnaire not found: {path}")
        return None
    return pd.read_excel(path)


def load_questionnaire_coding():
    path = QUESTIONNAIRE_DIR / "Questionnaire_Coding.xlsx"
    if not path.exists():
        print(f"[MISSING] Questionnaire coding not found: {path}")
        return None
    return pd.read_excel(path)
