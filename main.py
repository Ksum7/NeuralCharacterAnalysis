import json
import spacy
import load_model as load_model
import streamlit as st
import processing as processing
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="Personality Traits Detection")
st.title("Determining Personality Traits Using Neural Networks")

@st.cache(allow_output_mutation=True, show_spinner=False)
def load_models():
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_md")

    with open(f'./mrc2_dict.json', 'r') as f:
        word_data = json.load(f)

    fModel = load_model.load_model("./models_info.pth")
    return fModel, nlp, word_data

with st.spinner('Wait for the model to load'):
    fModel, nlp, word_data = load_models()

source_type = st.radio(
    "Source Type",
    ["Text", "YT video link"],
    horizontal=True
)
label_names = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']

def draw_polar_chart(data):
    ln = label_names.copy()
    lab = data

    df = pd.DataFrame(dict(r=lab, theta=ln))
    fig = px.line_polar(df, r='r', theta='theta', line_close=True)
    fig.update_traces(fill='toself')
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                tickmode='array',
                tickvals=[0, 20, 40, 60, 80, 100],
                ticktext=[tick + ' ' * 2 * len(tick) for tick in ['0%', '20% ', '40% ', '60% ', '80% ', '100%']],
                tickfont=dict(size=10, color='black', family='Arial'),
                tickangle=0,
                gridcolor='gray',
                range=[0, 100]
            ),
        ),
        width=600,
        margin=dict(l=100, r=100, t=20, b=20)
    )
    fig.update_xaxes(tickangle=0, tickmode='array', tickvals=[0, 20, 40, 60, 80, 100], ticktext=['0%', '20%', '40%', '60%', '80%', '100%'])
    st.plotly_chart(fig, use_container_width=False)

def big5_to_mbti(big5):
    res = ""
    res += "e" if big5[0] >= 0.5 else "i"
    res += "n" if big5[4] >= 0.5 else "s"
    res += "f" if big5[2] >= 0.5 else "t"
    res += "j" if big5[3] >= 0.5 else "p"
    return res.upper()

def print_info(outputs):
    columns = st.columns(len(label_names))

    for i, column in enumerate(columns):
        metric_value = round(outputs[i] * 100, 2)
        column.metric(label_names[i], f"{metric_value}%")

    mbti_type = big5_to_mbti(outputs)
    styled_text = f"<p style='font-size:24px; margin-bottom: 10px;'>Potential MBTI type: <span style='color:red;'>{mbti_type}</span></p>"
    st.markdown(styled_text, unsafe_allow_html=True)

    draw_polar_chart([o * 100 for o in outputs])

if source_type == "Text":
    try:
        text_input = st.text_area('Enter text', help="Input accepted in any language", value="")
        if st.button('Send'):
            if not text_input:
                raise ValueError("Text input is empty. Please enter some text.")
            with st.spinner('Processing...'):
                outputs = processing.evalLongText(fModel, nlp, word_data, text_input)
            print_info(outputs)      
    except ValueError as ve:
        st.error(str(ve))
    except Exception as e:
        print(e)
        st.error("An error occurred while processing the input. Please try again.")
else:
    try:
        url_input = st.text_input('Enter url')
        if st.button('Send'):
            if not url_input:
                raise ValueError("Url input is empty. Please enter url.")
            with st.spinner('Processing...'):
                outputs = processing.evalYT(fModel, nlp, word_data, url_input)
            print_info(outputs)
    except ValueError as ve:
        st.error(str(ve))
    except Exception as e:
        print(e)
        st.error("An error occurred while processing the input. Please try again.")
