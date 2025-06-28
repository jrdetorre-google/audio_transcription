from google.oauth2 import service_account
import json
from google import genai
from google.genai import types
import base64
import datetime # NUEVO: Importamos el módulo datetime

class AudioTranscription:
    def __init__(self, service_account_key_path):
        with open(service_account_key_path, 'r') as f:
            service_account_info = json.load(f)
            
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            # Puedes especificar los scopes explícitamente si lo deseas,
            # pero los roles de IAM ya asignados a la cuenta de servicio suelen ser suficientes
            scopes=['https://www.googleapis.com/auth/cloud-platform'] 
        )
        
        # Configurar el cliente
        self.client = genai.Client(
            vertexai=True,
            project="jrdetorre-genai-demo",
            location="global",
            credentials=credentials
        )


    def transcribe(self, audio_file_path, output_csv_file):
        
        msg1_audio1 = types.Part.from_uri(
            file_uri=audio_file_path,
            mime_type="audio/wav",
        )
        
        msg1_text1 = types.Part.from_text(text="""Convierte este audio correspondiente a una llamada a un centro de atención telefónica a texto. 
            Utiliza sólo información de esta llamada, no utilices nada más. No inventes nada. 
            Considera que puede haber mensajes muy irritados en la llamada. Si los encuentras, no te preocupes y sigue con las instrucciones que te he dado. 
            Estate muy atento a las transiciones. Puede haber secuencias muy rápidas en las que los participantes se interrumpan entre sí. 
            Ten en cuenta que puede haber silencios prolongados, por lo que no pares de transcripbir hasta el final del audio.
            Es importante que el orden de salida sea el orden real de la conversación y los timestamps sean correctos.
            Tiene que estar la transcripción completa, no puede faltar nada de lo que se ha dicho en la llamada.
            Devuelve el resultado en una tabla que contenga los siguientes campos, con una fila con cada mensaje comunicado por un hablante. 
            line_number: Código secuencial correspondiente a cada nueva línea 
            speaker: Hablante que interviene en cada nueva secuencia del mensaje. Si es una secuencia con silencio identifícalo con hablante \"Silencio\" 
            init_time: Tiempo de inicio de cada secuencia del mensaje, expresado con respecto al inicio de la llamada expresado en minutos:segundos 
            end_time: Tiempo de fin de cada secuencia del mensaje, expresado con respecto al inicio de la llamada expresado en minutos:segundos 
            duration: La duración de la secuencia. debería ser igual a Tiempo fin - Tiempo inicial expresado en minutos:segundos 
            language_code: Código iso del, por ejemplo es-es, es-co, en-en... 
            message: Texto escrito de la secuencia del mensaje. Si hay alguna secuencia con silencio introduce aquí alguna descripción de qué está pasando, si hay ruido de fondo, de qué tipo o cualquier mensaje descriptivo que consideres importante. No inventes, por favor. 
            tone. Emoción principal de la secuencia. Interesa principalmente Neutra, Irritada, Conciliadora, pero incluye cualquier otra que necesites 
            alert: Rellena este campo con \"Ojo\", si ves que hay palabras malsonantes, amenazas, una falta de respeto o la conversación es violenta. Déjalo vacío si no es el caso. No pongas otras cosas que no sea \"Ojo\" 
            pii: Incluye aquí los datos personales que identifiques en la secuencia. Puede ser una identidad con nombre y apellidos, una dirección, un identificador de cliente...             
            """)

        model = "gemini-2.5-flash"
#        model = "gemini-2.5-pro"
        contents = [
            types.Content(
            role="user",
            parts=[
                msg1_audio1,
                msg1_text1
            ]
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature = 1,
            top_p = 1,
            seed = 0,
            max_output_tokens = 65535,
            audio_timestamp=True,
            safety_settings = [types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
            ),types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
            ),types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
            ),types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
            )],
            system_instruction=[types.Part.from_text(text="""Eres un analista de audios, especialista en analizar audios de llamadas del call center de Gana Energia, compañia comercializadora de Luz y Gas.""")],
            thinking_config=types.ThinkingConfig(
                thinking_budget=-1,
            ),
            response_mime_type="application/json",
            response_schema=genai.types.Schema(
            type = genai.types.Type.OBJECT,
            required = ["line_number", "speaker","init_time","end_time","duration","language_code","message","tone","alert","pii"],
            properties = {
                "line_number": genai.types.Schema(
                    type = genai.types.Type.STRING,
                ),
                "speaker": genai.types.Schema(
                    type = genai.types.Type.STRING,
                ),
                "init_time": genai.types.Schema(
                    type = genai.types.Type.STRING,
                ),
                "end_time": genai.types.Schema(
                    type = genai.types.Type.STRING,
                ),
                "duration": genai.types.Schema(
                    type = genai.types.Type.STRING,
                ),
                "language_code": genai.types.Schema(
                    type = genai.types.Type.STRING,
                ),
                "message": genai.types.Schema(
                    type = genai.types.Type.STRING,
                ),
                "tone": genai.types.Schema(
                    type = genai.types.Type.STRING,
                ),
                "alert": genai.types.Schema(
                    type = genai.types.Type.STRING,
                ),
                "pii": genai.types.Schema(
                    type = genai.types.Type.STRING,
                ),
                },
            ),
        )

        print("Iniciando la transcripción batch...")
        start_time = datetime.datetime.now()
        print(f"Hora de inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        response = self.client.models.generate_content(
            model = model,
            contents = contents,
            config = generate_content_config,
        )

        end_time = datetime.datetime.now()
        print(f"Hora de fin:    {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Transcripción finalizada, escribimos a disco")

        duration = end_time - start_time
        print(f"Duración total: {duration}")
        print("-" * 40) # Separador para mayor claridad

        with open(output_csv_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        # Mensaje de confirmación en la consola
        print(f"La transcripción se ha guardado correctamente en el archivo: {output_csv_file}")

        
        
if __name__ == "__main__":
    audio_transcription = AudioTranscription("/home/admin_/vertex_user-jrdetorre-genai-demo-91dae877df6c.json")
    audio_transcription.transcribe("gs://gana_recordings/call.wav", "transcripcion_2506280830.json")
