from src.features import highlighted_suspicious_terms, risk_level


def test_risk_level_thresholds():
    assert risk_level(0.2) == "Low"
    assert risk_level(0.5) == "Medium"
    assert risk_level(0.9) == "High"


def test_highlighted_suspicious_terms_marks_keywords():
    text = highlighted_suspicious_terms("A secret miracle cure was exposed.")
    assert "**secret**" in text
    assert "**miracle**" in text
