import streamlit as st


def float_to_percent(score) -> str:
    # Ensure score is a float
    if type(score) == list:
        score = score[0]
        return score
    score = float(score)
    # Convert to percent and round
    percent = round(score * 100, 1)
    return f"{percent}%"
