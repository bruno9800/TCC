# 🌐 Cloudflare Tunnel Setup

Para expor sua API (`http://localhost:8000`) para a internet de forma segura e com URL fixa:

## 1. Obter o Token
1. Acesse o [Zero Trust Dashboard](https://one.dash.cloudflare.com/).
2. Vá em **Networks > Tunnels**.
3. Clique em **Create a Tunnel**.
4. Escolha **Cloudflared** e dê um nome (ex: `tcc-api`).
5. Copie o **Token** que aparecerá na tela de instalação (é uma string longa começando com `ey...`).

## 2. Configurar o Projeto
1. Abra o arquivo `.env`.
2. Adicione a variável `TUNNEL_TOKEN` com o valor copiado:
   ```bash
   TUNNEL_TOKEN=eyJhIjoi...
   ```

## 3. Configurar a Rota (Public Hostname)
1. No dashboard do Cloudflare, após salvar o tunnel, vá na aba **Public Hostname**.
2. Adicione uma rota:
   - **Subdomain**: `api` (ex: `api.seu-dominio.com`) ou deixe em branco para usar o domínio raiz.
   - **Domain**: Escolha seu domínio.
   - **Service**: `HTTP` -> `univasf-api:8000` (Importante: use o nome do container `univasf-api`).

## 4. Rodar
Reinicie os containers:
```bash
docker compose up -d
```

O container `univasf-tunnel` irá se conectar e sua API estará acessível no domínio configurado!
