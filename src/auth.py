from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from src.config import TCC_API_KEY

# Define o nome do header esperado
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    """
    Valida a API Key fornecida no header 'x-api-key'.
    Se TCC_API_KEY não estiver configurada no servidor, permite acesso livre (modo desenvolvimento inseguro).
    """
    if not TCC_API_KEY:
        # Se não houver senha configurada, deixa passar (CUIDADO em produção)
        return None

    if api_key == TCC_API_KEY:
        return api_key
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials"
    )
