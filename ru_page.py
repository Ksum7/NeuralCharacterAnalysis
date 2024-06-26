import streamlit as st
def draw_page(load_models, processing, print_info):
    st.set_page_config(layout="wide", page_title="Определение личностных характеристик")
    st.title("Определение личностных характеристик человека с использованием нейронных сетей")

    st.sidebar.title("Навигация")
    selection = st.sidebar.radio("Перейти", ["О работе", "Предвзятость, риски и ограничения", "Рекомендации", "Модель"])

    if selection == "О работе":
        st.markdown(
            """
            ## О работе

            Эта работа демонстрирует возможности модели в обработке текста для прогнозирования личностных характеристик по пятифакторной модели и индикатору типа Майерс-Бриггс (MBTI).
            
            Модель универсальна и способна принимать текстовые данные на любом языке.
            
            Кроме того, она может анализировать транскрибированные данные с YouTube-видео, что показывает возможность её использования за пределами обработки простых текстовых данных
           
            ## Пятифакторная модель личности

            Пятифакторная модель личности, также известная как модель "Большой пятёрки" (Big Five) и модель OCEAN, является таксономией или классификацией личностных черт. Пять факторов включают:

            - **Открытость опыту:** Изобретательность/Любознательность vs. Консервативность/Осторожность
            - **Добросовестность:** Эффективность/Организованность vs. Лёгкость/Беспечность
            - **Экстраверсия:** Общительность/Энергичность vs. Скрытность/Замкнутость
            - **Доброжелательность:** Дружелюбие/Сострадание vs. Конфликтность/Отстранённость
            - **Нейротизм:** Чувствительность/Нервозность vs. Уверенность/Спокойствие

            ## Типология Майерс-Бриггс

            Типология Майерс-Бриггс (MBTI) - это интроспективный самотест, указывающий на различия в психологических предпочтениях в том, как люди воспринимают мир и принимают решения.

            |     | Субъективное    | Объективное    |
            |-----|----------------|----------------|
            | Дедуктивное | Интуиция/Ощущение | Интроверсия/Экстраверсия |
            | Индуктивное | Чувство/Мышление | Восприятие/Суждение |

            Комбинации четырёх пар признаков приводят к 16 возможным комбинациям, также известным как типы. 16 типов обычно обозначаются аббревиатурой из четырёх букв — первыми буквами каждого из четырёх признаков (за исключением интуиции, которая обозначается буквой "N", чтобы отличить её от интроверсии). Например:

            - **ESTJ:** экстраверсия (E), ощущение (S), мышление (T), суждение (J)
            - **INFP:** интроверсия (I), интуиция (N), чувство (F), восприятие (P)
            """
        )
    elif selection == "Предвзятость, риски и ограничения":
        st.write(
            """
            Модель предсказания личностных характеристик, как и любая модель машинного обучения, имеет определенные ограничения и потенциальные предвзятости, которые следует учитывать:
            """
        )
        st.markdown(
            """
            - **Ограниченный контекст**: 
                Модель делает прогнозы только на основе введенного текста и может не охватить полный контекст личности человека. Важно учитывать, что личностные черты формируются под влиянием множества факторов, выходящих за рамки текстового выражения.

            - **Обобщение**: 
                Модель прогнозирует личностные черты на основе шаблонов, изученных на определенном наборе данных. Её производительность может варьироваться при применении к людям из разных демографических или культурных групп, которые недостаточно представлены в обучающем наборе данных.

            - **Этические соображения**: 
                Модели предсказания личностных характеристик должны использоваться ответственно, с пониманием того, что личностные черты не определяют ценность или способности человека. Важно избегать несправедливых суждений или дискриминации людей на основе предсказанных личностных черт.

            - **Проблемы конфиденциальности**: 
                Модель использует введенный пользователем текст, который может содержать конфиденциальную или личную информацию. Пользователи должны быть осторожны при предоставлении личных данных и обеспечивать безопасность своих данных.

            - **Ложные срабатывания/Пропуски**: 
                Прогнозы модели могут не всегда точно соответствовать реальным личностным чертам человека. Возможны как ложные срабатывания (предсказание черты, которой нет), так и пропуски (непредсказание присутствующей черты).
            """
        )
    elif selection == "Рекомендации":
        st.write(
            """
            Чтобы смягчить риски и ограничения, связанные с моделями предсказания личностных характеристик, предлагаются следующие рекомендации:
            """
        )
        st.markdown(
            """
            - **Информированность и обучение**: 
                Пользователи должны быть информированы о ограничениях и возможных предвзятостях модели. Пропагандировать понимание того, что личностные черты сложны и не могут быть полностью охвачены одной моделью или анализом текста.

            - **Избегать стереотипов и дискриминации**: 
                Пользователи должны быть осторожны при вынесении суждений или принятии решений только на основе предсказанных личностных черт. Прогнозы личностных характеристик не должны использоваться для дискриминации людей или поддержания стереотипов.

            - **Интерпретация в контексте**: 
                Интерпретировать прогнозы модели в соответствующем контексте и учитывать дополнительную информацию о человеке помимо его введенного текста.

            - **Конфиденциальность и безопасность данных**: 
                Обеспечить безопасное обращение с данными пользователей и соблюдение норм конфиденциальности. Пользователи должны быть осведомлены о предоставляемой информации и проявлять осторожность при обмене личными данными.

            - **Продвижение этического использования**: 
                Поощрять ответственное использование моделей предсказания личностных характеристик и препятствовать их злоупотреблению или вредному применению.
            """
        )
    elif selection == "Модель":
        st.markdown(
            """
            Модель обрабатывает текст для предоставления прогнозов как по пятифакторной модели личности, так и по индикатору типа Майерс-Бриггс (MBTI). 
            Её универсальность распространяется на обработку текстовых вводов на любом языке. 
            Кроме того, этот пример иллюстрирует, как модель может использоваться для анализа транскрибированного контента с YouTube-видео, 
            демонстрируя её универсальность за пределами обработки простых текстовых данных.
            """
        )

        with st.spinner('Ждите, модель загружается'):
            fModel, nlp, word_data = load_models()

        source_type = st.radio(
            "Тип источника",
            ["Текст", "Ссылка на видео YouTube"],
            horizontal=True
        )

        if source_type == "Текст":
            try:
                text_input = st.text_area('Введите текст', help="Принимаются вводы на любом языке", value="")
                if st.button('Отправить'):
                    if not text_input:
                        raise ValueError("Текстовое поле пустое. Пожалуйста, введите текст.")
                    with st.spinner('Обработка...'):
                        outputs = processing.evalLongText(fModel, nlp, word_data, text_input)
                    print_info(outputs)      
            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                print(e)
                st.error("Произошла ошибка при обработке ввода. Попробуйте снова.")
        else:
            try:
                url_input = st.text_input('Введите URL')
                if st.button('Отправить'):
                    if not url_input:
                        raise ValueError("Поле URL пустое. Пожалуйста, введите URL.")
                    with st.spinner('Обработка...'):
                        outputs = processing.evalYT(fModel, nlp, word_data, url_input)
                    print_info(outputs)
            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                print(e)
                st.error("Произошла ошибка при обработке ввода. Попробуйте снова.")
                
    def change_lang():
        l = st.session_state.langselect
        if l == "Английский" or l == "English":
            lang = 'en'
        elif l == "Русский" or l == "Russian":
            lang = 'ru'
        if lang != st.query_params['language']:
            st.query_params['language'] = lang
            # st.rerun()

    lang = st.sidebar.selectbox("Язык", ["Русский", "Английский"], key='langselect', on_change=change_lang)
