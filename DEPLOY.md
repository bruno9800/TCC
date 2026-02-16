# 🚀 Guia de Deploy em VPS (Ubuntu)

Este guia descreve como configurar e rodar o projeto em uma VPS usando Docker.

## 1. Pré-requisitos na VPS
Acesse sua VPS via SSH e instale o Docker e Docker Compose:

```bash
# Atualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Dar permissão ao usuário atual (evita usar sudo sempre)
sudo usermod -aG docker $USER
newgrp docker

# Verificar instalação
docker --version
docker compose version
```

## 2. Configuração Inicial do Projeto

1. **Clone o repositório**:
   ```bash
   git clone https://github.com/bruno9800/TCC.git
   cd TCC
   ```

2. **Crie o arquivo `.env`**:
   Copie o exemplo e edite com sua chave da OpenAI.
   ```bash
   cp .env.example .env
   nano .env
   ```
   > **Importante**: Mantenha `CHROMA_HOST=chromadb` e `CHROMA_PORT=8000` para deploy.

## 3. Rodando o Projeto

### Iniciar os Serviços
```bash
docker compose up -d --build
```

### Indexar Documentos (Primeira vez apenas)
Seus documentos PDF devem estar na pasta `regimentos_estatutos_resolucoes/` antes de rodar este passo.
A indexação persiste no volume `chroma_data`, então você só precisa rodar uma vez ou quando adicionar novos arquivos.

```bash
docker compose run --rm api python scripts/run_indexing.py
```

## 4. Atualizando o Projeto (Deploy Contínuo)
Para baixar a versão mais recente do código e atualizar os containers:

```bash
git pull                   # Baixa alterações do GitHub
docker compose up -d --build --remove-orphans # Reconstrói e reinicia
```

## 5. Rodando o Frontend (Streamlit)
O Streamlit também pode rodar via Docker. Se ainda não tiver um serviço definido para ele, adicione ao `docker-compose.yml` ou use o script de automação abaixo.

---

# 🤖 Automação de Deploy

Crie um script chamado `deploy.sh` na raiz do projeto para facilitar atualizações futuras:

1. Crie o arquivo:
   ```bash
   nano deploy.sh
   ```

2. Cole o conteúdo:
   ```bash
   #!/bin/bash
   
   echo "🚀 Iniciando Deploy..."
   
   # 1. Puxar código atualizado
   echo "📥 Baixando atualizações do Git..."
   git pull
   
   # 2. Subir containers (Build se necessário)
   echo "🐳 Construindo e subindo containers..."
   docker compose up -d --build --remove-orphans
   
   # 3. Limpar imagens antigas (opcional, economiza espaço)
   echo "🧹 Limpando imagens não utilizadas..."
   docker image prune -f
   
   echo "✅ Deploy concluído com sucesso!"
   ```

3. Dê permissão de execução:
   ```bash
   chmod +x deploy.sh
   ```

4. Para atualizar o projeto no futuro, basta rodar:
   ```bash
   ./deploy.sh
   ```
