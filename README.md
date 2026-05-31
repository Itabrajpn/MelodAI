# MelodAI PRO MultiEngine FIX

Correções:
- HeyGen v3 com `type: image`, `image: {type: asset_id}`, `audio_asset_id`.
- Upload correto por `POST /v3/assets`.
- Painel mostra se Wav2Lip, SadTalker e MuseTalk foram encontrados.
- Mensagem de erro explica o caminho exato que falta.

## Rodar

```bat
cd C:\Users\msile\Desktop\MelodAI_PRO_MultiEngine_FIX
C:\Users\msile\AppData\Local\Programs\Python\Python310\python.exe -m pip install -r requirements.txt
copy .env.example .env
notepad .env
C:\Users\msile\AppData\Local\Programs\Python\Python310\python.exe app.py
```

Abra:

```text
http://localhost:5000
```

## Para motores locais

Copie as pastas completas para dentro desta pasta:

```text
Wav2Lip
SadTalker
MuseTalk
```

Ou edite o `.env` com os caminhos completos.
