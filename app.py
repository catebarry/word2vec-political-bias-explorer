# This file (UI components and debugging) was created with help from ChatGPT 5.0 (see AI Assistance Statement)
import streamlit as st
from PrecalculatedBiasCalculator import PrecalculatedBiasCalculator
from parse_sentence import parse_sentence
import pandas as pd
import json
import altair as alt
from os import path

# --- Constants ---
MAX_BIAS = 3.0 #0.7
NEUTRAL_THRESHOLD = 0.1

# Load calculator with caching
@st.cache_resource
def load_calculator():
    return PrecalculatedBiasCalculator()

calc = load_calculator()

# --- Helper Functions ---

def norm_bias(bias):
    """Normalize bias to [0, 1] scale, capped at MAX_BIAS."""
    if bias is None:
        return 0.0
    return min(abs(bias), MAX_BIAS) / MAX_BIAS

def bias_arrow_length(bias):
    """Return arrow length as percentage (0-100)."""
    return 60 * norm_bias(bias)

def bias_color(bias):
    if bias is None or norm_bias(bias) < NEUTRAL_THRESHOLD:
        return "#CCCCCC"
    
    base_color = "#3F8EAA" if is_dem_bias(bias) else "#AA3F3F"
    norm = norm_bias(bias)
    
    # Lighten based on bias strength
    lightening = int((1 - norm) * 120)
    return lighten_darken_color(base_color, lightening)

def lighten_darken_color(color_hex, percent):
    """Lighten or darken a hex color."""
    color_hex = color_hex.lstrip('#')
    num = int(color_hex, 16)
    r = max(0, min(255, (num >> 16) + percent))
    g = max(0, min(255, ((num >> 8) & 0x00FF) + percent))
    b = max(0, min(255, (num & 0x0000FF) + percent))
    return f"#{r:02x}{g:02x}{b:02x}"

def is_neutral(bias):
    """Check if bias is neutral."""
    if bias is None:
        return True
    return abs(norm_bias(bias)) < NEUTRAL_THRESHOLD

def is_dem_bias(bias):
    """Check if bias leans Democrat/left."""
    return not is_neutral(bias) and bias < 0 # CHANGE DEPENDING ON WHICH WAY IS DEM

def bias_text(bias):
    """Return human-readable bias label."""
    if is_neutral(bias):
        return "neutral"
    
    party = "Democrat" if is_dem_bias(bias) else "Republican"
    norm = norm_bias(bias)
    
    if norm < 0.3:
        amount = "slight"
    elif norm < 0.6:
        amount = "moderate"
    else:
        amount = "strong"
    
    return f"{amount} {party} bias"

def arrow_html(bias):
    """Generate HTML for directional arrow."""
    if is_neutral(bias):
        return None
    
    color = bias_color(bias)
    length = bias_arrow_length(bias)
    direction = "left" if is_dem_bias(bias) else "right"
    # Create a centered arrow pointing left or right using svg
    if direction == "right":
        arrow_svg = f"""
        <svg width="140" height="24" style="display: block;">
            <line x1="70" y1="12" x2="{70 + length}" y2="12" stroke="{color}" stroke-width="4" stroke-linecap="round"/>
            <polygon points="{70 + length + 5},12 {70 + length - 4},8 {70 + length - 4},16" fill="{color}"/>
        </svg>
        """
    else:  # left
        arrow_svg = f"""
        <svg width="140" height="24" style="display: block;">
            <line x1="70" y1="12" x2="{70 - length}" y2="12" stroke="{color}" stroke-width="4" stroke-linecap="round"/>
            <polygon points="{70 - length - 5},12 {70 - length + 4},8 {70 - length + 4},16" fill="{color}"/>
        </svg>
        """
    
    return arrow_svg


# --- Streamlit UI ---

st.set_page_config(page_title="Word2Vec Political Bias Explorer", layout="centered")

st.markdown(
    "<h1 style='text-align: center;'>Word2Vec Political Bias Explorer</h1>",
    unsafe_allow_html=True
)
st.markdown("Type a word or phrase and press **Enter** to see its political association.")

# Search form
with st.form("search_form"):
    sentence = st.text_input(
        "",
        key="sentence_input",
        placeholder="e.g., healthcare, freedom, stimulus",
        label_visibility="collapsed"
    )
    submitted = st.form_submit_button("Enter", use_container_width=True)

# Results display
if submitted and sentence.strip():
    results = []
    
    # Parse sentence into tokens
    parsed_tokens = parse_sentence(sentence)
    
    for token_obj in parsed_tokens:
        token_text = token_obj["text"]
        bias_val = calc.detect_bias(token_text)
        
        # Check if any parts should be skipped
        skipped = any(p.get("skip", False) for p in token_obj.get("parts", []))
        
        results.append({
            "token": token_text,
            "bias": bias_val,
            "skipped": skipped,
            "parts": token_obj.get("parts", [])
        })
    
    # Display results
    if results:
        st.markdown("---")
        
        for result in results:
            token = result["token"]
            bias = result["bias"]
            
            # Create columns for layout
            col1, col2, col3 = st.columns([2, 3, 2])
            
            with col1:
                st.write(f"**{token}**")
            
            with col2:
                arrow = arrow_html(bias)
                if arrow:
                    st.markdown(arrow, unsafe_allow_html=True)
                else:
                    st.write("")
            
            with col3:
                # Compact display: bias label + numbers on one tight line
                if bias is None:
                    bias_label = f"<span>{bias_text(bias)}</span>"
                    bias_numbers = "<span style='font-size:12px; color:#006400;'>norm: N/A&nbsp;&nbsp;raw: N/A</span>"
                else:
                    bias_label = f"<span>{bias_text(bias)}</span>"
                    bias_numbers = (
                        f"<span style='font-size:12px; color:#006400;'>"
                        f"norm: {round(norm_bias(bias),3)}&nbsp;&nbsp;"
                        f"raw: {round(bias,3)}</span>"
                    )

                st.markdown(
                    f"<div style='line-height:1.0; margin:0; padding:0;'>{bias_label}<br>{bias_numbers}</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("No tokens found in the input.")

elif submitted and not sentence.strip():
    st.warning("Please enter a word or phrase.")


# --- How it works --- 
st.markdown("---")
st.subheader("How it works")
st.markdown(
    """
This tool projects words from a pretrained GoogleNews Word2Vec embedding onto a **political axis**. 
It uses methodologies from [Word2Vec Gender Bias Explorer](https://chanind.github.io/word2vec-gender-bias-explorer/#/) 
adapted for political bias, and takes inspiration from [Studying Political Bias via Word Embeddings](https://dl.acm.org/doi/10.1145/3366424.3383560).

The bias is computed by creating a political axis from seed word pairs like (democrat, republican), (liberal, conservative), etc., 
then projecting each word onto this axis using PCA, 
and scaling to [-1, 1] where negative scores indicate Democratic/left-leaning association and positive scores indicate Republican/right-leaning association.
The arrows show bias direction and strength, pointing left for Democratic bias and right for Republican bias.
    """
)

# --- Curated words continuum ---

DATA = path.join(path.dirname(__file__), "data", "biases.json")

# Edit this list to test words!
TEST_WORDS = [
    "democrat", "republican", 
    "CNN", "Obama",
    "immigration", "invasion", "refugee", "illegal",
    "healthcare", "capitalism", "socialism",
    "taxes", "welfare",
    "traditional", "charity", "union", "security",
    "right", "left", "freedom",
    "church", "military", "veterans", "guns",
    "business", "capital", "markets", "Fox",
    "defense", "oil", "energy",
    "coal", "industry", "corporation",
    "police", "law_enforcement",
]

with open(DATA, "r") as f:
    biases = json.load(f)

rows = []
for w in TEST_WORDS:
    raw = biases.get(w, None)
    if raw is None:
        norm_signed = None
    else:
        norm_signed = max(-1.0, min(1.0, float(raw) / MAX_BIAS))
    rows.append({"word": w, "raw": None if raw is None else float(raw), "norm_signed": norm_signed})

df_test = pd.DataFrame(rows)
# label side using normalized signed threshold (NEUTRAL_THRESHOLD relative to MAX_BIAS)
threshold_signed = NEUTRAL_THRESHOLD  # since norm is in [0,1], NEUTRAL_THRESHOLD fits directly
df_test["side"] = df_test["norm_signed"].apply(
    lambda x: "N/A" if x is None else ("Neutral" if abs(x) < threshold_signed else ("Democrat" if x < 0 else "Republican"))
)

# sort for plotting: lowest norm_signed first (Democrat left, Republican right)
chart_df = df_test.sort_values("norm_signed", ascending=True, na_position="first").reset_index(drop=True)

st.markdown("---")
st.subheader("Curated words — normalized political bias continuum")

# Build chart using normalized signed score as the x-axis (range -1..1)
bars = alt.Chart(chart_df[chart_df["norm_signed"].notna()]).mark_bar().encode(
    y=alt.Y("word:N", sort=alt.EncodingSortField(field="norm_signed", op="min"), title=None, axis=alt.Axis(labelFontSize=10, labelLimit=220, labelAngle=0, labelOverlap=False)),
    x=alt.X("norm_signed:Q", title="Normalized bias (negative = Democrat, positive = Republican)", scale=alt.Scale(domain=[-1, 1])),
    color=alt.condition(alt.datum.norm_signed < 0, alt.value("#3F8EAA"), alt.value("#AA3F3F")),
    tooltip=[
        alt.Tooltip("word:N", title="word"),
        alt.Tooltip("raw:Q", format=".4f", title="raw"),
        alt.Tooltip("norm_signed:Q", format=".3f", title="norm_signed"),
        alt.Tooltip("side:N", title="side")
    ])
    
zero_rule = alt.Chart(pd.DataFrame({"x":[0]})).mark_rule(color="black").encode(x="x:Q")
chart = (bars + zero_rule).properties(height=20 * max(3, len(chart_df)), width=800)
st.altair_chart(chart, use_container_width=True)

