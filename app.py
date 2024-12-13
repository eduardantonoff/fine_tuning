import streamlit as st
from transformers import pipeline

# Загружаем модель Hugging Face
@st.cache_resource
def load_model():
    return pipeline('text-generation', model='gpt2')

generator = load_model()

# Заголовок приложения
st.title("Генератор отзывов")

# Поля для ввода данных
category = st.selectbox(
    "Категория места:",
    ["Ресторан", "Отель", "Магазин", "Достопримечательность", "Другое"]
)

rating = st.radio(
    "Средний рейтинг:",
    ["Отрицательный", "Нейтральный", "Положительный"]
)

keywords = st.text_input("Ключевые слова (через запятую):")

# Кнопка для генерации
if st.button("Сгенерировать отзывы"):
    # Формируем начальный текст
    input_text = f"Категория: {category}. Рейтинг: {rating}. Ключевые слова: {keywords}. Отзыв: "

    # Генерация отзывов
    outputs = generator(input_text, max_new_tokens=50, num_return_sequences=3)

    # Отображаем результаты
    st.subheader("Сгенерированные отзывы:")
    for i, output in enumerate(outputs):
        st.write(f"{i + 1}. {output['generated_text']}")
