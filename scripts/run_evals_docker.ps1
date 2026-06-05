# Roda avaliacao DeepEval dentro do conteiner Streamlit (RAG/OLAP/ML nos volumes Docker).
# Uso (na raiz do repo):
#   .\scripts\run_evals_docker.ps1
#   .\scripts\run_evals_docker.ps1 --limit 5 --category rag
#
# Rebuild obrigatorio apos adicionar/alterar arquivos em apps/streamlit/evals/:
#   docker compose build streamlit

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$EvalArgs
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host ">> Rebuild da imagem streamlit (inclui evals/)..." -ForegroundColor Cyan
docker compose build streamlit
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ">> Recriando conteiner streamlit (aplica imagem nova)..." -ForegroundColor Cyan
docker compose up -d streamlit
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($EvalArgs.Count -eq 0) {
    $EvalArgs = @("--require-ready", "--limit", "3")
}

Write-Host ">> Executando evals: python evals/run_assistente_eval.py $($EvalArgs -join ' ')" -ForegroundColor Cyan
docker compose exec -e LANGFUSE_ENABLED=0 streamlit python evals/run_assistente_eval.py @EvalArgs
exit $LASTEXITCODE
