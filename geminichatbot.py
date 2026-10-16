from google import genai
from google.genai import errors
from google.genai import types
client= genai.Client()

chat= client.chats.create(
     model="gemini-3.5-flash-lite",
     config=types.GenerateContentConfig(
         system_instruction ="You are an AI Engineering tutor. Explain concepts clearly, with short answers and simple Python examples."
     )
)

while True:
    question = input("You: ")

    if question.strip().lower() in ["exit", "quit", "bye"]:
        print("Chatbot: Goodbye!")
        break
    try:
        response= chat.send_message(question)
        #response= client.models.generate_content(
        #    model="gemini-3.5-flash",
            #contents="Explain Artificial Intelligence in 2 sentences."
        #    contents=question
        #)

        print("Chatbot:", response.text)

    except errors.ServerError as e:
        if e.code == 503:
            print("Chatbot: Gemini is busy. Please try again.")
        else:
            print("Chatbot: Server error:", e)