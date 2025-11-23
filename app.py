# streamlit_app.py
from PcaBiasCalculator import PcaBiasCalculator
from PrecalculatedBiasCalculator import PrecalculatedBiasCalculator
from parse_sentence import parse_sentence
import pandas as pd

# Load the fast precalculated calculator (reads data/biases.json)
@st.cache_resource
def load_calculator():
    return PrecalculatedBiasCalculator()

calc = load_calculator()

st.set_page_config(page_title="Political Bias Explorer", layout="wide")
st.title("Political Bias Explorer (Streamlit)")

st.markdown(
    "Type a word or phrase. Tokens (including compounds like 'asylum_seeker') "
    "will be shown with their bias score (negative = left/Democrat, positive = right/Republican)."
)

query = st.text_input("Enter a word or phrase", value="", max_chars=200)

neutral_words = ["is", "was", "who", "what", "where", "the", "it"]  # same as repo

if query:
    objs = parse_sentence(query)
    rows = []
    for obj in objs:
        token_text = obj["text"]
        bias_val = calc.detect_bias(token_text)
        # Show N/A if none
        rows.append({
            "token": token_text,
            "bias": None if bias_val is None else float(bias_val),
            "skipped": any(
                (p["skip"] for p in obj.get("parts", []))
            )
        })

    # render table: keep the token order from parse_sentence
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No tokens found.")
    else:
        # Format bias column nicely
        def fmt(x):
            return "N/A" if x is None else f"{x:+.2f}"

        df_display = df.copy()
        df_display["bias"] = df_display["bias"].apply(fmt)
        st.table(df_display[["token", "bias"]])

        # show most extreme token (by absolute score) if available
        numeric = [r for r in rows if r["bias"] is not None]
        if numeric:
            most_extreme = max(numeric, key=lambda r: abs(r["bias"]))
            st.markdown(
                f"**Most extreme token:** `{most_extreme['token']}` — score {most_extreme['bias']:+.2f}"
            )
        else:
            st.info("No scored tokens found in reduced vocab.")
