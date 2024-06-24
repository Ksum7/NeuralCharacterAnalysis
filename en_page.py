import streamlit as st
def draw_page(load_models, processing, print_info):
    st.set_page_config(layout="wide", page_title="Personality Traits Detection")
    st.title("Determining Personality Traits Using Neural Networks")

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["About", "Bias, Risks, and Limitations", "Recommendations", "Model"])

    if selection == "About":
        st.markdown(
            """
            ## About

            This work is dedicated to showcasing the capabilities of a specific model in processing text to provide predictions for both 
                the Big Five personality traits and the Myers-Briggs Type Indicator (MBTI). 

            It demonstrates the model's versatility by accommodating text inputs in any language. 

            Furthermore, it highlights the model's capability to analyze transcribed content from YouTube videos, 
                thereby illustrating its versatility beyond handling straightforward text inputs.

            ## The Big Five Personality Model

            The Big Five personality traits, also known as the five-factor model (FFM) 
            and the OCEAN model, is a taxonomy, or grouping, for personality traits. 
            The five factors are:

            - **Openness to experience:** Inventive/Curious vs. Consistent/Cautious
            - **Conscientiousness:** Efficient/Organized vs. Easy-going/Careless
            - **Extraversion:** Outgoing/Energetic vs. Solitary/Reserved
            - **Agreeableness:** Friendly/Compassionate vs. Challenging/Detached
            - **Neuroticism:** Sensitive/Nervous vs. Secure/Confident

            ## The Myers–Briggs Type Indicator

            The Myers–Briggs Type Indicator (MBTI) is an introspective self-report questionnaire indicating differing psychological preferences in how people perceive the world and make decisions.

            |     | Subjective     | Objective      |
            |-----|----------------|----------------|
            | Deductive | Intuition/Sensing | Introversion/Extraversion |
            | Inductive | Feeling/Thinking | Perception/Judging |

            The combinations as four pairs of preferences lead to 16 possible combinations aka types. The 16 types are typically referred to by an abbreviation of four letters—the initial letters of each of their four type preferences (except in the case of intuition, which uses the abbreviation "N" to distinguish it from introversion). For instance:

            - **ESTJ:** extraversion (E), sensing (S), thinking (T), judgment (J)
            - **INFP:** introversion (I), intuition (N), feeling (F), perception (P)

            """
        )
    elif selection == "Bias, Risks, and Limitations":
        st.write(
            """
            The personality prediction model, like any machine learning model, has certain limitations and potential biases that should be taken into account:
            """
        )
        st.markdown(
            """
            - **Limited Context**: 
                The model makes predictions based on input text alone and may not capture the full context of an individual's personality. It is important to consider that personality traits are influenced by various factors beyond textual expression.

            - **Generalization**: 
                The model predicts personality traits based on patterns learned from a specific dataset. Its performance may vary when applied to individuals from different demographic or cultural backgrounds not well represented in the training data.

            - **Ethical Considerations**: 
                Personality prediction models should be used responsibly, with an understanding that personality traits do not determine a person's worth or abilities. It is important to avoid making unfair judgments or discriminating against individuals based on predicted personality traits.

            - **Privacy Concerns**: 
                The model relies on user-provided input text, which may contain sensitive or personal information. Users should exercise caution when sharing personal details and ensure the security of their data.

            - **False Positives/Negatives**: 
                The model's predictions may not always align perfectly with an individual's actual personality traits. It is possible for the model to generate false positives (predicting a trait that is not present) or false negatives (missing a trait that is present).
            """
        )
    elif selection == "Recommendations":
        st.write(
            """
            To mitigate risks and limitations associated with personality prediction models, the following recommendations are suggested:
            """
        )
        st.markdown(
            """
            - **Awareness and Education**: 
                Users should be informed about the limitations and potential biases of the model. Promote understanding that personality traits are complex and cannot be fully captured by a single model or text analysis.

            - **Avoid Stereotyping and Discrimination**: 
                Users should be cautious about making judgments or decisions solely based on predicted personality traits. Personality predictions should not be used to discriminate against individuals or perpetuate stereotypes.

            - **Interpret with Context**: 
                Interpret the model's predictions in the appropriate context and consider additional information about an individual beyond their input text.

            - **Data Privacy and Security**: 
                Ensure that user data is handled securely and with respect to privacy regulations. Users should be aware of the information they provide and exercise caution when sharing personal details.

            - **Promote Ethical Use**: 
                Encourage responsible use of personality prediction models and discourage misuse or harmful applications.
            """
        )
    elif selection == "Model":
        st.markdown(
            """
            The model processes text to deliver predictions for both the Big Five personality traits and the Myers-Briggs Type Indicator (MBTI). 
            Its versatility extends to accommodating text inputs in any language. 
            Additionally, this example illustrates how the model can be used to analyze transcribed content from YouTube videos, 
            showcasing its versatility beyond processing simple text inputs.
            """
        )

        with st.spinner('Wait for the model to load'):
            fModel, nlp, word_data = load_models()

        source_type = st.radio(
            "Source Type",
            ["Text", "YT video link"],
            horizontal=True
        )

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
    def change_lang():
        l = st.session_state.langselect
        if l == "Английский" or l == "English":
            lang = 'en'
        elif l == "Русский" or l == "Russian":
            lang = 'ru'
        if lang != st.query_params['language']:
            st.query_params['language'] = lang
            # st.rerun()
    lang = st.sidebar.selectbox("Language", ["English", "Russian"], key='langselect', on_change=change_lang)

    