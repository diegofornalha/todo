from fastapi.middleware.cors import CORSMiddleware

def configure_cors(app):
    """
    Configura o CORS para o servidor LangGraph.
    """
    origins = [
        "https://smith.langchain.com",
        "http://localhost:8123",
        "http://127.0.0.1:8123",
        "http://0.0.0.0:8123",
        "http://192.168.1.18:8123",
        "https://*.ngrok.io",
        "https://*.ngrok-free.app",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Permite todas as origens em desenvolvimento
        allow_credentials=True,
        allow_methods=["*"],  # Permite todos os métodos
        allow_headers=["*"],  # Permite todos os headers
        expose_headers=["*"],
        max_age=3600,
        allow_origin_regex=None,
        expose_headers_list=["*"],
        preflight_max_age=3600,
    )
    return app 