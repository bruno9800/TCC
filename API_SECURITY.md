# 🔐 Segurança da API (API Key)

Sua API agora está protegida! Qualquer requisição para `/chat` precisa incluir o header `x-api-key`.

## 1. Configurar a Senha
No seu arquivo `.env`, adicione:
```bash
TCC_API_KEY=sua-senha-super-secreta
```
> Se você não definir essa variável, a API continuará aberta (modo inseguro).

## 2. Como usar (Clientes)

### Streamlit (Frontend)
O frontend já foi atualizado para ler a variável `TCC_API_KEY` do ambiente.
Basta garantir que o `.env` ou as variáveis de ambiente onde o Streamlit roda tenham a chave.

### Curl / Postman
Adicione o header:
```bash
curl -X POST "https://api.facilitaagro.app/chat/" \
     -H "Content-Type: application/json" \
     -H "x-api-key: sua-senha-super-secreta" \
     -d '{"message": "Olá"}'
```

### Python (Requests/Httpx)
```python
import httpx

httpx.post(
    "https://api.facilitaagro.app/chat/",
    headers={"x-api-key": "sua-senha-super-secreta"},
    json={"message": "Olá"}
)
```
