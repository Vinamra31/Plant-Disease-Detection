call C:\Users\vinam\anaconda3\Scripts\activate.bat ml_env

start cmd /k "uvicorn main:app --reload"
start cmd /k "streamlit run app.py"